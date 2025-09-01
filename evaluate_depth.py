from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import datasets
import networks

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

# --- added imports for pruning, benchmarking, and reporting ---
import time
import json
import csv
from datetime import datetime
from pathlib import Path
import torch.nn as nn
import torch.nn.utils.prune as prune
# extra imports for system metrics and plotting
try:
    import psutil
except Exception:  # fallback if psutil isn't available
    psutil = None
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:
    plt = None
try:
    import pynvml  # NVIDIA GPU utilization
    _pynvml_available = True
except Exception:
    _pynvml_available = False
# --------------------------------------------------------------

splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


# ========================= NEW HELPERS =========================

def _iter_prunable_modules(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            yield m


def apply_pruning(model, prune_type: str, amount: float):
    """Apply pruning in-place.
    prune_type in {"unstructured", "structured_l1"}; amount in [0.0, 0.5]
    """
    if amount <= 0.0:
        return
    for m in _iter_prunable_modules(model):
        if not hasattr(m, "weight"):
            continue
        if prune_type == "unstructured":
            prune.l1_unstructured(m, name="weight", amount=amount)
        elif prune_type == "structured_l1":
            # prune entire output channels/filters by L1 norm
            dim = 0  # output channels for Conv2d/Linear
            prune.ln_structured(m, name="weight", amount=amount, n=1, dim=dim)
        else:
            raise ValueError(f"Unknown prune_type: {prune_type}")
        # Make pruning permanent: leaves zeros in weights, keeps tensor shapes
        prune.remove(m, "weight")


def count_param_stats(*models):
    total_params = 0
    nonzero_params = 0
    for model in models:
        for p in model.parameters():
            numel = p.numel()
            total_params += numel
            nonzero_params += torch.count_nonzero(p).item()
    sparsity = 0.0 if total_params == 0 else (1.0 - (nonzero_params / total_params))
    return total_params, nonzero_params, sparsity


def get_fresh_models(encoder_state, depth_state, num_layers, device):
    encoder = networks.ResnetEncoder(num_layers, False)
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

    # encoder_state contains extra keys like 'height'/'width'; filter by model keys
    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_state.items() if k in model_dict})
    depth_decoder.load_state_dict(depth_state)

    encoder.to(device).eval()
    depth_decoder.to(device).eval()
    return encoder, depth_decoder


def forward_predict_and_benchmark(dataloader, encoder, depth_decoder, opt, device):
    """Run forward passes over dataloader and return disparities and timing.

    Returns:
        pred_disps (np.ndarray)
        avg_latency_ms_per_image (float)
        fps (float)
        memory_usage (float)   # MB
        gpu_usage (float)      # % utilization, NaN if unavailable
    """
    pred_disps = []
    total_time = 0.0
    total_images = 0

    # Track memory/usage
    proc = psutil.Process(os.getpid()) if psutil is not None else None
    peak_cpu_mem = 0  # bytes

    peak_gpu_mem_mb = float('nan')
    avg_gpu_util = float('nan')
    gpu_util_sum = 0.0
    gpu_util_samples = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        if _pynvml_available:
            try:
                pynvml.nvmlInit()
                cuda_index = torch.cuda.current_device()
                handle = pynvml.nvmlDeviceGetHandleByIndex(int(cuda_index))
            except Exception:
                handle = None
        else:
            handle = None
    else:
        handle = None

    # Warmup for CUDA
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            input_color = data[("color", 0, 0)].to(device, non_blocking=True)
            if opt.post_process:
                input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)
            _ = depth_decoder(encoder(input_color))
            if i >= 1:  # two warmup iters
                break

    with torch.no_grad():
        for data in dataloader:
            input_color = data[("color", 0, 0)].to(device, non_blocking=True)

            images_out = input_color.shape[0]
            if opt.post_process:
                input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            output = depth_decoder(encoder(input_color))
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()

            total_time += (t1 - t0)
            total_images += images_out  # latency per original image

            pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
            pred_disp = pred_disp.detach().cpu()[:, 0].numpy()

            if opt.post_process:
                N = pred_disp.shape[0] // 2
                pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

            pred_disps.append(pred_disp)

            # Update CPU peak memory (bytes)
            if proc is not None:
                try:
                    rss = proc.memory_info().rss
                    peak_cpu_mem = max(peak_cpu_mem, rss)
                except Exception:
                    pass

            # Update GPU memory/util
            if device.type == "cuda":
                try:
                    peak_bytes = torch.cuda.max_memory_allocated(device)
                    peak_gpu_mem_mb = max(peak_gpu_mem_mb, peak_bytes / (1024**2)) if not np.isnan(peak_gpu_mem_mb) else peak_bytes / (1024**2)
                except Exception:
                    pass
                if handle is not None:
                    try:
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        gpu_util_sum += float(util.gpu)
                        gpu_util_samples += 1
                    except Exception:
                        pass

    pred_disps = np.concatenate(pred_disps) if len(pred_disps) > 0 else np.array([])

    if total_images == 0 or total_time == 0:
        avg_latency_ms = float("nan")
        fps = float("nan")
    else:
        avg_latency_ms = (total_time / total_images) * 1000.0
        fps = total_images / total_time

    # Choose a single memory_usage number for the table: prefer GPU if available
    if device.type == "cuda" and not np.isnan(peak_gpu_mem_mb):
        memory_usage = float(peak_gpu_mem_mb)  # MB
    else:
        memory_usage = float(peak_cpu_mem / (1024**2)) if peak_cpu_mem else float('nan')

    avg_gpu_util = float(gpu_util_sum / gpu_util_samples) if gpu_util_samples > 0 else float('nan')

    return pred_disps, avg_latency_ms, fps, memory_usage, avg_gpu_util


def evaluate_from_disps_and_gt(pred_disps, gt_depths, opt):
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    errors = []
    ratios = []

    # Determine scaling behavior (same as original)
    if opt.eval_stereo:
        disable_median_scaling = True
        pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        disable_median_scaling = False
        pred_depth_scale_factor = getattr(opt, "pred_depth_scale_factor", 1)

    for i in range(pred_disps.shape[0]):
        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / np.maximum(pred_disp, 1e-12)

        if opt.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)
        else:
            mask = gt_depth > 0

        pred_depth = pred_depth[mask]
        gt_depth_masked = gt_depth[mask]

        pred_depth *= pred_depth_scale_factor
        if not disable_median_scaling:
            ratio = np.median(gt_depth_masked) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth_masked, pred_depth))

    mean_errors = np.array(errors).mean(0) if len(errors) else np.array([float("nan")]*7)

    if not disable_median_scaling and len(ratios):
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    return mean_errors


def write_results(results, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV with requested columns
    csv_path = out_dir / "results.csv"
    fieldnames = [
        "prune_type", "amount", "params_total", "params_nonzero",
        "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3",
        "latency_ms", "fps", "memory usage", "gpu usage"
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    # JSON (store all keys)
    json_path = out_dir / "results.json"
    with json_path.open("w") as f:
        json.dump(results, f, indent=2)

    # Markdown table with requested columns
    md_path = out_dir / "results.md"
    with md_path.open("w") as f:
        f.write("# Pruning and Evaluation Results\n\n")
        f.write("| prune_type | amount | params_total | params_nonzero | abs_rel | sq_rel | rmse | rmse_log | a1 | a2 | a3 | latency_ms | fps | memory usage | gpu usage |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in results:
            f.write(
                "| {prune_type} | {amount:.0%} | {params_total} | {params_nonzero} | "
                "{abs_rel:.3f} | {sq_rel:.3f} | {rmse:.3f} | {rmse_log:.3f} | {a1:.3f} | {a2:.3f} | {a3:.3f} | "
                "{latency_ms:.2f} | {fps:.2f} | {mem:.2f} | {gpu:.1f} |\n".format(
                    prune_type=r.get("prune_type", ""),
                    amount=r.get("amount", 0.0),
                    params_total=r.get("params_total", 0),
                    params_nonzero=r.get("params_nonzero", 0),
                    abs_rel=r.get("abs_rel", float("nan")),
                    sq_rel=r.get("sq_rel", float("nan")),
                    rmse=r.get("rmse", float("nan")),
                    rmse_log=r.get("rmse_log", float("nan")),
                    a1=r.get("a1", float("nan")),
                    a2=r.get("a2", float("nan")),
                    a3=r.get("a3", float("nan")),
                    latency_ms=r.get("latency_ms", float("nan")),
                    fps=r.get("fps", float("nan")),
                    mem=r.get("memory usage", float("nan")),
                    gpu=r.get("gpu usage", float("nan")),
                )
            )

    return {"csv": str(csv_path), "json": str(json_path), "md": str(md_path)}


def plot_results(results, out_dir: Path):
    if plt is None:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data grouped by prune_type
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        groups[r.get("prune_type", "")].append(r)

    saved = {}

    for ptype, rows in groups.items():
        # sort by amount
        rows = sorted(rows, key=lambda x: x.get("amount", 0.0))
        amounts = [r.get("amount", 0.0) * 100 for r in rows]  # percent

        # Evaluation metrics: plot small-range metrics together with scaling (x100)
        small_keys = ["abs_rel", "sq_rel", "rmse_log", "a1", "a2", "a3"]
        fig, ax = plt.subplots(figsize=(10, 6))
        for key in small_keys:
            ys = [r.get(key, np.nan) for r in rows]
            ys_scaled = [y * 100.0 if y == y else np.nan for y in ys]  # scale by 100
            ax.plot(amounts, ys_scaled, marker='o', label=key)
        ax.set_xlabel('Pruning amount (%)')
        ax.set_ylabel('Scaled metric value (x100)')
        ax.set_title(f'Evaluation metrics (small-range) vs pruning ({ptype})')
        ax.grid(True, ls='--', alpha=0.4)
        ax.legend(ncol=2)
        eval_small_path = out_dir / f"evaluation_metrics_small_{ptype}.png"
        fig.tight_layout()
        fig.savefig(eval_small_path, dpi=150)
        plt.close(fig)
        saved[f"evaluation_small_{ptype}"] = str(eval_small_path)

        # RMSE plotted separately
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ys_rmse = [r.get("rmse", np.nan) for r in rows]
        ax2.plot(amounts, ys_rmse, marker='o', color='tab:red', label='rmse')
        ax2.set_xlabel('Pruning amount (%)')
        ax2.set_ylabel('RMSE')
        ax2.set_title(f'RMSE vs pruning ({ptype})')
        ax2.grid(True, ls='--', alpha=0.4)
        ax2.legend()
        eval_rmse_path = out_dir / f"evaluation_rmse_{ptype}.png"
        fig2.tight_layout()
        fig2.savefig(eval_rmse_path, dpi=150)
        plt.close(fig2)
        saved[f"evaluation_rmse_{ptype}"] = str(eval_rmse_path)

        # Hardware metrics in separate figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        hw = [
            ("latency_ms", "Latency (ms)"),
            ("fps", "FPS"),
            ("memory usage", "Memory usage (MB)"),
            ("gpu usage", "GPU usage (%)"),
        ]
        for ax, (k, label) in zip(axes.ravel(), hw):
            ys = [r.get(k, np.nan) for r in rows]
            ax.plot(amounts, ys, marker='s')
            ax.set_title(label)
            ax.set_xlabel('Pruning amount (%)')
            ax.grid(True, ls='--', alpha=0.4)
        fig.suptitle(f'Hardware metrics vs pruning ({ptype})')
        hw_path = out_dir / f"hardware_metrics_{ptype}.png"
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        fig.savefig(hw_path, dpi=150)
        plt.close(fig)
        saved[f"hardware_{ptype}"] = str(hw_path)

        # --- Combined two-subplot figure per prune_type ---
        # For 'structured_l1' show 'sq_rel' separately; for 'unstructured' show 'rmse' separately
        special_key = "sq_rel" if ptype == "structured_l1" else "rmse"
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

        # Left subplot: special metric
        y_special = [r.get(special_key, np.nan) for r in rows]
        ax_left.plot(amounts, y_special, marker='o', color='tab:red', label=special_key)
        ax_left.set_title(f'{special_key.upper()} vs pruning ({ptype})')
        ax_left.set_xlabel('Pruning amount (%)')
        ax_left.set_ylabel(special_key.upper())
        ax_left.grid(True, ls='--', alpha=0.4)
        ax_left.legend()

        # Right subplot: other metrics grouped (scaled x100)
        group_keys = ["abs_rel", "sq_rel", "rmse_log", "a1", "a2", "a3"]
        group_plot_keys = [k for k in group_keys if k != special_key]
        for key in group_plot_keys:
            ys = [r.get(key, np.nan) for r in rows]
            ys_scaled = [y * 100.0 if y == y else np.nan for y in ys]
            ax_right.plot(amounts, ys_scaled, marker='o', label=key)
        ax_right.set_title('Other metrics (scaled x100)')
        ax_right.set_xlabel('Pruning amount (%)')
        ax_right.set_ylabel('Scaled metric value (x100)')
        ax_right.grid(True, ls='--', alpha=0.4)
        ax_right.legend(ncol=1)

        combo_path = out_dir / f"evaluation_metrics_combo_{ptype}.png"
        fig.tight_layout()
        fig.savefig(combo_path, dpi=150)
        plt.close(fig)
        saved[f"evaluation_combo_{ptype}"] = str(combo_path)

    return saved


# ====================== END NEW HELPERS ========================


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set, with pruning sweeps and benchmarking.
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    # Only run pruning/eval when we are generating predictions from a model
    if opt.ext_disp_to_eval is not None:
        # Fallback to original behavior for external disparity evaluation
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))
            pred_disps = pred_disps[eigen_to_benchmark_ids]

        if opt.no_eval:
            print("-> Evaluation disabled. Done.")
            quit()

        if opt.eval_split == 'benchmark':
            print("-> No ground truth is available for the KITTI benchmark, so not evaluating. Done.")
            quit()

        gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
        gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]

        print("-> Evaluating external disparities")
        if opt.eval_stereo:
            print("   Stereo evaluation - disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
            opt.disable_median_scaling = True
            opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
        else:
            print("   Mono evaluation - using median scaling")

        errors = []
        ratios = []
        for i in range(pred_disps.shape[0]):
            gt_depth = gt_depths[i]
            gt_height, gt_width = gt_depth.shape[:2]
            pred_disp = pred_disps[i]
            pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
            pred_depth = 1 / np.maximum(pred_disp, 1e-12)

            if opt.eval_split == "eigen":
                mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
                crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                                 0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
                crop_mask = np.zeros(mask.shape)
                crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
                mask = np.logical_and(mask, crop_mask)
            else:
                mask = gt_depth > 0

            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]

            pred_depth *= getattr(opt, "pred_depth_scale_factor", 1)
            if not getattr(opt, "disable_median_scaling", False):
                ratio = np.median(gt_depth) / np.median(pred_depth)
                ratios.append(ratio)
                pred_depth *= ratio

            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

            errors.append(compute_errors(gt_depth, pred_depth))

        if not getattr(opt, "disable_median_scaling", False):
            ratios = np.array(ratios)
            med = np.median(ratios)
            print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

        mean_errors = np.array(errors).mean(0)
        print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
        print("\n-> Done!")
        return

    # ----------------- Full pruning + evaluation path -----------------

    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)

    print("-> Loading weights from {}".format(opt.load_weights_folder))

    filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

    encoder_state = torch.load(encoder_path, map_location="cpu")
    depth_state = torch.load(decoder_path, map_location="cpu")

    # Dataset/Dataloader
    feed_height = int(encoder_state['height'])
    feed_width = int(encoder_state['width'])
    dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                       feed_height, feed_width,
                                       [0], 4, is_train=False)
    dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=opt.num_workers,
                            pin_memory=True, drop_last=False)

    # Output dirs
    pruned_ckpt_root = Path(opt.load_weights_folder) / "test_pruned_models"
    results_root = Path(opt.load_weights_folder) / "pruning_eval_results" / datetime.now().strftime("%Y%m%d-%H%M%S")
    results_root.mkdir(parents=True, exist_ok=True)

    # Ground truth depths if available
    gt_depths = None
    if opt.eval_split != 'benchmark':
        gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
        gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]

    # Scaling behavior (printed once)
    if opt.eval_stereo:
        print("   Stereo evaluation - disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    prune_types = ["unstructured", "structured_l1"]
    amounts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    results = []

    # Evaluate across pruning settings
    print("-> Computing predictions with size {}x{}".format(feed_width, feed_height))

    for prune_type in prune_types:
        for amount in amounts:
            # Fresh models per setting
            encoder, depth_decoder = get_fresh_models(encoder_state, depth_state, opt.num_layers, device)

            # Apply pruning
            if amount > 0:
                apply_pruning(encoder, prune_type, amount)
                apply_pruning(depth_decoder, prune_type, amount)

            # Save pruned weights neatly
            subdir = (
                pruned_ckpt_root / prune_type / f"{int(amount*100)}pct" if amount > 0
                else pruned_ckpt_root / "original"
            )
            subdir.mkdir(parents=True, exist_ok=True)
            torch.save(encoder.state_dict(), str(subdir / "encoder.pth"))
            torch.save(depth_decoder.state_dict(), str(subdir / "depth.pth"))

            # Count parameters/sparsity
            params_total, params_nonzero, sparsity_ratio = count_param_stats(encoder, depth_decoder)

            # Forward + timing + hardware
            pred_disps, latency_ms, fps, mem_usage, gpu_util = forward_predict_and_benchmark(
                dataloader, encoder, depth_decoder, opt, device
            )

            # Evaluate metrics if GT available
            if gt_depths is not None and pred_disps.size > 0:
                mean_errors = evaluate_from_disps_and_gt(pred_disps, gt_depths, opt)
                abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = mean_errors.tolist()
            else:
                abs_rel = sq_rel = rmse = rmse_log = a1 = a2 = a3 = float('nan')

            # Record results with requested columns
            res = {
                "prune_type": prune_type,
                "amount": amount,
                "params_total": int(params_total),
                "params_nonzero": int(params_nonzero),
                "abs_rel": float(abs_rel),
                "sq_rel": float(sq_rel),
                "rmse": float(rmse),
                "rmse_log": float(rmse_log),
                "a1": float(a1),
                "a2": float(a2),
                "a3": float(a3),
                "latency_ms": float(latency_ms),
                "fps": float(fps),
                "memory usage": float(mem_usage) if mem_usage == mem_usage else float('nan'),
                "gpu usage": float(gpu_util) if gpu_util == gpu_util else float('nan'),
                # extra info kept in JSON
                "device": device.type,
                "batch_size": dataloader.batch_size,
                "eval_split": opt.eval_split,
                "weights_dir": str(subdir),
                "sparsity_ratio": float(sparsity_ratio),
            }
            results.append(res)

            # Short console line per setting
            print("[{} | {:>2}%] FPS: {:.2f}  Latency(ms): {:.2f}  abs_rel: {:.3f}  a1: {:.3f}  mem(MB): {:.2f}  gpu(%): {}".format(
                prune_type, int(amount*100), fps, latency_ms, res["abs_rel"], res["a1"], res["memory usage"],
                "{:.1f}".format(res["gpu usage"]) if res["gpu usage"] == res["gpu usage"] else "na"
            ))

    # Write summary tables
    paths = write_results(results, results_root)

    # Plots
    plot_paths = plot_results(results, results_root)

    print("\n-> Results saved:")
    for k, v in paths.items():
        print(f"   {k}: {v}")
    if plot_paths:
        for k, v in plot_paths.items():
            print(f"   {k}: {v}")

    print("\n-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
