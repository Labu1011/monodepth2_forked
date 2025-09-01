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

# --- imports for benchmarking, reporting, and visualization ---
import time
import json
import csv
from datetime import datetime
from pathlib import Path
import torch.nn as nn
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


def count_param_stats(*models):
    """Count total and nonzero parameters across models"""
    total_params = 0
    nonzero_params = 0
    for model in models:
        for p in model.parameters():
            numel = p.numel()
            total_params += numel
            nonzero_params += torch.count_nonzero(p).item()
    sparsity = 0.0 if total_params == 0 else (1.0 - (nonzero_params / total_params))
    return total_params, nonzero_params, sparsity


def get_model_size_mb(model):
    """Calculate model size in memory (MB)"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    model_size = (param_size + buffer_size) / (1024 ** 2)  # Convert to MB
    return model_size


def get_file_size_mb(file_path):
    """Get file size on disk in MB"""
    if os.path.exists(file_path):
        return os.path.getsize(file_path) / (1024 ** 2)
    return 0.0


def load_models(weights_folder, num_layers, device, is_quantized=False):
    """Load encoder and depth decoder models"""
    if is_quantized:
        encoder_path = os.path.join(weights_folder, "encoder_quantized.pth")
        decoder_path = os.path.join(weights_folder, "depth_quantized.pth")
    else:
        encoder_path = os.path.join(weights_folder, "encoder.pth")
        decoder_path = os.path.join(weights_folder, "depth.pth")
    
    if not os.path.exists(encoder_path) or not os.path.exists(decoder_path):
        return None, None
    
    # Load state dicts
    encoder_state = torch.load(encoder_path, map_location="cpu")
    depth_state = torch.load(decoder_path, map_location="cpu")
    
    # Create models
    encoder = networks.ResnetEncoder(num_layers, False)
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)
    
    # Load weights - handle the case where encoder_state might have extra keys
    if 'height' in encoder_state and 'width' in encoder_state:
        # Original format with metadata
        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_state.items() if k in model_dict})
    else:
        # Pure state dict
        encoder.load_state_dict(encoder_state)
    
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
    """Evaluate predicted disparities against ground truth"""
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
    """Write results to CSV, JSON, and Markdown files"""
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV with requested columns
    csv_path = out_dir / "quantization_results.csv"
    fieldnames = [
        "model_type", "model_size_mb", "memory_size_mb", "params_total", "params_nonzero",
        "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3",
        "latency_ms", "fps", "memory_usage", "gpu_usage"
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    # JSON (store all keys)
    json_path = out_dir / "quantization_results.json"
    with json_path.open("w") as f:
        json.dump(results, f, indent=2)

    # Markdown table
    md_path = out_dir / "quantization_results.md"
    with md_path.open("w") as f:
        f.write("# Quantization Evaluation Results\n\n")
        f.write("| model_type | model_size_mb | memory_size_mb | params_total | params_nonzero | abs_rel | sq_rel | rmse | rmse_log | a1 | a2 | a3 | latency_ms | fps | memory_usage | gpu_usage |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in results:
            f.write(
                "| {model_type} | {model_size:.2f} | {memory_size:.2f} | {params_total} | {params_nonzero} | "
                "{abs_rel:.3f} | {sq_rel:.3f} | {rmse:.3f} | {rmse_log:.3f} | {a1:.3f} | {a2:.3f} | {a3:.3f} | "
                "{latency_ms:.2f} | {fps:.2f} | {mem:.2f} | {gpu:.1f} |\n".format(
                    model_type=r.get("model_type", ""),
                    model_size=r.get("model_size_mb", 0.0),
                    memory_size=r.get("memory_size_mb", 0.0),
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
                    mem=r.get("memory_usage", float("nan")),
                    gpu=r.get("gpu_usage", float("nan")),
                )
            )

    return {"csv": str(csv_path), "json": str(json_path), "md": str(md_path)}


def plot_comparison(results, out_dir: Path):
    """Create comparison plots between original and quantized models"""
    if plt is None:
        return None
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Separate original and quantized results
    original_result = next((r for r in results if r["model_type"] == "original"), None)
    quantized_result = next((r for r in results if r["model_type"] == "quantized"), None)
    
    if not original_result or not quantized_result:
        print("Warning: Missing original or quantized results for comparison")
        return None
    
    saved_plots = {}
    
    # 1. Evaluation metrics comparison (bar chart)
    eval_metrics = ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(eval_metrics))
    width = 0.35
    
    original_values = [original_result.get(m, float('nan')) for m in eval_metrics]
    quantized_values = [quantized_result.get(m, float('nan')) for m in eval_metrics]
    
    bars1 = ax.bar(x - width/2, original_values, width, label='Original', alpha=0.8)
    bars2 = ax.bar(x + width/2, quantized_values, width, label='Quantized', alpha=0.8)
    
    ax.set_xlabel('Evaluation Metrics')
    ax.set_ylabel('Metric Value')
    ax.set_title('Evaluation Metrics: Original vs Quantized Models')
    ax.set_xticks(x)
    ax.set_xticklabels(eval_metrics, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    
    eval_metrics_path = out_dir / "evaluation_metrics_comparison.png"
    fig.tight_layout()
    fig.savefig(eval_metrics_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved_plots["evaluation_metrics"] = str(eval_metrics_path)
    
    # 2. Performance metrics comparison (bar chart)
    perf_metrics = ["latency_ms", "fps", "memory_usage", "model_size_mb"]
    perf_labels = ["Latency (ms)", "FPS", "Memory Usage (MB)", "Model Size (MB)"]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for i, (metric, label) in enumerate(zip(perf_metrics, perf_labels)):
        ax = axes[i]
        
        orig_val = original_result.get(metric, float('nan'))
        quant_val = quantized_result.get(metric, float('nan'))
        
        bars = ax.bar(['Original', 'Quantized'], [orig_val, quant_val], 
                     color=['tab:blue', 'tab:orange'], alpha=0.8)
        
        ax.set_ylabel(label)
        ax.set_title(f'{label} Comparison')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')
        
        # Calculate improvement/degradation
        if not np.isnan(orig_val) and not np.isnan(quant_val) and orig_val != 0:
            if metric in ["latency_ms", "memory_usage", "model_size_mb"]:
                # Lower is better
                improvement = ((orig_val - quant_val) / orig_val) * 100
                direction = "reduction" if improvement > 0 else "increase"
            else:
                # Higher is better (fps)
                improvement = ((quant_val - orig_val) / orig_val) * 100
                direction = "improvement" if improvement > 0 else "degradation"
            
            ax.text(0.5, 0.95, f'{abs(improvement):.1f}% {direction}', 
                   transform=ax.transAxes, ha='center', va='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    
    fig.suptitle('Performance Metrics: Original vs Quantized Models')
    perf_metrics_path = out_dir / "performance_metrics_comparison.png"
    fig.tight_layout()
    fig.savefig(perf_metrics_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved_plots["performance_metrics"] = str(perf_metrics_path)
    
    # 3. Combined summary plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Top-left: Key evaluation metrics
    key_eval = ["abs_rel", "rmse", "a1"]
    key_eval_labels = ["Abs Rel", "RMSE", "δ < 1.25"]
    x = np.arange(len(key_eval))
    width = 0.35
    
    orig_vals = [original_result.get(m, float('nan')) for m in key_eval]
    quant_vals = [quantized_result.get(m, float('nan')) for m in key_eval]
    
    ax1.bar(x - width/2, orig_vals, width, label='Original', alpha=0.8, color='tab:blue')
    ax1.bar(x + width/2, quant_vals, width, label='Quantized', alpha=0.8, color='tab:orange')
    ax1.set_xlabel('Evaluation Metrics')
    ax1.set_ylabel('Metric Value')
    ax1.set_title('Key Evaluation Metrics')
    ax1.set_xticks(x)
    ax1.set_xticklabels(key_eval_labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Top-right: Speed comparison
    speed_metrics = ["fps", "latency_ms"]
    speed_labels = ["FPS", "Latency (ms)"]
    
    for i, (metric, label) in enumerate(zip(speed_metrics, speed_labels)):
        orig_val = original_result.get(metric, float('nan'))
        quant_val = quantized_result.get(metric, float('nan'))
        
        bars = ax2.bar([f'{label}\n(Original)', f'{label}\n(Quantized)'], [orig_val, quant_val], 
                      color=['tab:blue', 'tab:orange'], alpha=0.8, width=0.4)
        
        if i == 0:  # FPS - position bars side by side
            bars[0].set_x(0.1)
            bars[1].set_x(0.5)
        else:  # Latency - position next to FPS bars
            bars[0].set_x(1.1)
            bars[1].set_x(1.5)
    
    ax2.set_ylabel('Value')
    ax2.set_title('Speed Performance')
    ax2.grid(True, alpha=0.3)
    
    # Bottom-left: Memory and model size
    orig_mem = original_result.get("memory_usage", float('nan'))
    quant_mem = quantized_result.get("memory_usage", float('nan'))
    orig_size = original_result.get("model_size_mb", float('nan'))
    quant_size = quantized_result.get("model_size_mb", float('nan'))
    
    bars = ax3.bar(['Memory Usage\n(Original)', 'Memory Usage\n(Quantized)', 
                   'Model Size\n(Original)', 'Model Size\n(Quantized)'], 
                  [orig_mem, quant_mem, orig_size, quant_size],
                  color=['tab:blue', 'tab:orange', 'tab:blue', 'tab:orange'], alpha=0.8)
    
    ax3.set_ylabel('Size (MB)')
    ax3.set_title('Memory and Model Size')
    ax3.grid(True, alpha=0.3)
    
    # Bottom-right: Summary text
    ax4.axis('off')
    summary_text = "Quantization Summary\n\n"
    
    # Calculate improvements
    metrics_summary = [
        ("Model Size", orig_size, quant_size, "reduction"),
        ("Memory Usage", orig_mem, quant_mem, "reduction"),
        ("Latency", original_result.get("latency_ms", float('nan')), 
         quantized_result.get("latency_ms", float('nan')), "reduction"),
        ("FPS", original_result.get("fps", float('nan')), 
         quantized_result.get("fps", float('nan')), "improvement"),
        ("Abs Rel Error", original_result.get("abs_rel", float('nan')), 
         quantized_result.get("abs_rel", float('nan')), "increase"),
        ("RMSE", original_result.get("rmse", float('nan')), 
         quantized_result.get("rmse", float('nan')), "increase"),
        ("δ < 1.25", original_result.get("a1", float('nan')), 
         quantized_result.get("a1", float('nan')), "decrease"),
    ]
    
    for name, orig, quant, change_type in metrics_summary:
        if not np.isnan(orig) and not np.isnan(quant) and orig != 0:
            if change_type in ["reduction", "increase"]:
                change_pct = ((orig - quant) / orig) * 100
                direction = "↓" if change_pct > 0 else "↑"
            else:  # improvement or decrease
                change_pct = ((quant - orig) / orig) * 100
                direction = "↑" if change_pct > 0 else "↓"
            
            summary_text += f"{name}: {direction} {abs(change_pct):.1f}%\n"
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
    
    fig.suptitle('Quantization Evaluation: Comprehensive Comparison', fontsize=16)
    summary_path = out_dir / "quantization_summary.png"
    fig.tight_layout()
    fig.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved_plots["summary"] = str(summary_path)
    
    return saved_plots


def evaluate_quantized(opt):
    """Evaluate both original and quantized models and compare them"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)

    print("-> Loading weights from {}".format(opt.load_weights_folder))

    # Check if quantized models exist
    encoder_quant_path = os.path.join(opt.load_weights_folder, "encoder_quantized.pth")
    depth_quant_path = os.path.join(opt.load_weights_folder, "depth_quantized.pth")
    
    if not os.path.exists(encoder_quant_path) or not os.path.exists(depth_quant_path):
        print(f"ERROR: Quantized models not found!")
        print(f"Looking for: {encoder_quant_path}")
        print(f"Looking for: {depth_quant_path}")
        return

    # Setup dataset
    filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
    
    # Load original model to get feed dimensions
    encoder_orig_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    if os.path.exists(encoder_orig_path):
        encoder_state = torch.load(encoder_orig_path, map_location="cpu")
        feed_height = int(encoder_state['height'])
        feed_width = int(encoder_state['width'])
    else:
        # Default dimensions if original not available
        feed_height, feed_width = 192, 640
        print(f"Using default dimensions: {feed_width}x{feed_height}")

    dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                       feed_height, feed_width,
                                       [0], 4, is_train=False)
    dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=opt.num_workers,
                            pin_memory=True, drop_last=False)

    # Output directory
    results_root = Path(opt.load_weights_folder) / "quantization_eval_results" / datetime.now().strftime("%Y%m%d-%H%M%S")
    results_root.mkdir(parents=True, exist_ok=True)

    # Ground truth depths if available
    gt_depths = None
    if opt.eval_split != 'benchmark':
        gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
        if os.path.exists(gt_path):
            gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]

    # Scaling behavior (printed once)
    if opt.eval_stereo:
        print("   Stereo evaluation - disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    results = []

    print("-> Computing predictions with size {}x{}".format(feed_width, feed_height))

    # Evaluate both original and quantized models
    model_configs = [
        ("original", False),
        ("quantized", True)
    ]

    for model_type, is_quantized in model_configs:
        print(f"\n-> Evaluating {model_type} model...")
        
        # Load models
        encoder, depth_decoder = load_models(opt.load_weights_folder, opt.num_layers, device, is_quantized)
        
        if encoder is None or depth_decoder is None:
            print(f"Failed to load {model_type} models, skipping...")
            continue

        # Count parameters and model size
        params_total, params_nonzero, sparsity_ratio = count_param_stats(encoder, depth_decoder)
        
        # Get file sizes on disk (this shows the actual storage reduction for quantized models)
        if is_quantized:
            encoder_path = os.path.join(opt.load_weights_folder, "encoder_quantized.pth")
            decoder_path = os.path.join(opt.load_weights_folder, "depth_quantized.pth")
        else:
            encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
            decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
        
        file_size_mb = get_file_size_mb(encoder_path) + get_file_size_mb(decoder_path)
        memory_size_mb = get_model_size_mb(encoder) + get_model_size_mb(depth_decoder)

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

        # Record results
        res = {
            "model_type": model_type,
            "model_size_mb": float(file_size_mb),  # File size on disk (shows quantization compression)
            "memory_size_mb": float(memory_size_mb),  # Memory footprint when loaded
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
            "memory_usage": float(mem_usage) if mem_usage == mem_usage else float('nan'),
            "gpu_usage": float(gpu_util) if gpu_util == gpu_util else float('nan'),
            # extra info
            "device": device.type,
            "batch_size": dataloader.batch_size,
            "eval_split": opt.eval_split,
            "sparsity_ratio": float(sparsity_ratio),
            "is_quantized": is_quantized,
        }
        results.append(res)

        # Print summary for this model
        print(f"[{model_type}] File Size: {file_size_mb:.2f}MB  Memory: {memory_size_mb:.2f}MB  FPS: {fps:.2f}  Latency: {latency_ms:.2f}ms  abs_rel: {abs_rel:.3f}  a1: {a1:.3f}")

    # Write results
    if len(results) > 0:
        paths = write_results(results, results_root)
        plot_paths = plot_comparison(results, results_root)

        print(f"\n-> Results saved to: {results_root}")
        for k, v in paths.items():
            print(f"   {k}: {v}")
        
        if plot_paths:
            print("\n-> Plots saved:")
            for k, v in plot_paths.items():
                print(f"   {k}: {v}")

        # Print comparison summary
        if len(results) == 2:
            orig_res = next(r for r in results if r["model_type"] == "original")
            quant_res = next(r for r in results if r["model_type"] == "quantized")
            
            print(f"\n-> Quantization Impact Summary:")
            print(f"   File Size: {orig_res['model_size_mb']:.2f}MB → {quant_res['model_size_mb']:.2f}MB ({((orig_res['model_size_mb'] - quant_res['model_size_mb']) / orig_res['model_size_mb'] * 100):.1f}% reduction)")
            print(f"   Memory Size: {orig_res['memory_size_mb']:.2f}MB → {quant_res['memory_size_mb']:.2f}MB ({((orig_res['memory_size_mb'] - quant_res['memory_size_mb']) / orig_res['memory_size_mb'] * 100):.1f}% change)")
            print(f"   FPS: {orig_res['fps']:.2f} → {quant_res['fps']:.2f} ({((quant_res['fps'] - orig_res['fps']) / orig_res['fps'] * 100):+.1f}%)")
            print(f"   Latency: {orig_res['latency_ms']:.2f}ms → {quant_res['latency_ms']:.2f}ms ({((orig_res['latency_ms'] - quant_res['latency_ms']) / orig_res['latency_ms'] * 100):+.1f}%)")
            print(f"   abs_rel: {orig_res['abs_rel']:.3f} → {quant_res['abs_rel']:.3f} ({((quant_res['abs_rel'] - orig_res['abs_rel']) / orig_res['abs_rel'] * 100):+.1f}%)")
            print(f"   a1: {orig_res['a1']:.3f} → {quant_res['a1']:.3f} ({((quant_res['a1'] - orig_res['a1']) / orig_res['a1'] * 100):+.1f}%)")

    print("\n-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate_quantized(options.parse())
