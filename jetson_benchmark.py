#!/usr/bin/env python3
"""
Jetson Nano TX2 Inference Benchmark for Monodepth2 ONNX Models

This script benchmarks original and quantized ONNX models on Jetson Nano TX2,
measuring performance metrics like FPS, latency, GPU and memory usage, and
visualizing the results for comparison.

Requirements (Python 3.6):
- onnxruntime
- numpy
- opencv-python
- matplotlib
- jetson-stats (for Jetson-specific GPU/memory monitoring)

Note: This script is specifically designed for Jetson Nano TX2 with Python 3.6.
"""

import os
import time
import json
import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union

import numpy as np
import cv2

# Conditional imports for Jetson-specific monitoring
try:
    import jtop  # jetson-stats package for Jetson monitoring
    JTOP_AVAILABLE = True
except ImportError:
    JTOP_AVAILABLE = False
    print("Warning: jetson-stats not available. Install with: pip install jetson-stats")

# ONNX Runtime
try:
    import onnxruntime as ort
    print(f"ONNX Runtime version: {ort.__version__}")
    print(f"Available providers: {ort.get_available_providers()}")
except ImportError:
    raise ImportError("ONNX Runtime is required. Install with: pip install onnxruntime")

# Optional - will be used for plotting if available
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark ONNX models on Jetson Nano TX2"
    )
    
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="./models/mono+stereo_640x192",
        help="Path to the directory containing ONNX models"
    )
    
    parser.add_argument(
        "--image_path", 
        type=str, 
        default="./assets/test_image.jpg",
        help="Path to test image or directory of images"
    )
    
    parser.add_argument(
        "--width", 
        type=int, 
        default=640,
        help="Input image width"
    )
    
    parser.add_argument(
        "--height", 
        type=int, 
        default=192,
        help="Input image height"
    )
    
    parser.add_argument(
        "--num_warmup", 
        type=int, 
        default=5,
        help="Number of warmup iterations"
    )
    
    parser.add_argument(
        "--num_iter", 
        type=int, 
        default=50,
        help="Number of benchmark iterations"
    )
    
    parser.add_argument(
        "--results_dir", 
        type=str, 
        default="./jetson_benchmark_results",
        help="Directory to save benchmark results"
    )
    
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1,
        help="Batch size for inference"
    )
    
    parser.add_argument(
        "--use_cuda", 
        action="store_true",
        help="Use CUDA execution provider if available"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug output for model compatibility"
    )
    
    return parser.parse_args()


class JetsonMonitor:
    """Monitor Jetson Nano TX2 hardware metrics."""
    
    def __init__(self):
        self.available = JTOP_AVAILABLE
        self.jetson = None
        self.stats = []
        
    def start(self):
        """Start monitoring."""
        if not self.available:
            return False
        
        try:
            self.jetson = jtop.jtop()
            self.jetson.start()
            self.stats = []
            return True
        except Exception as e:
            print(f"Error starting Jetson monitoring: {e}")
            self.available = False
            return False
    
    def sample(self):
        """Sample current hardware stats."""
        if not self.available or self.jetson is None:
            return
        
        try:
            if self.jetson.ok():
                stats = {
                    'timestamp': time.time(),
                    'gpu': self.jetson.gpu.get('gpu', 0),  # GPU utilization %
                    'gpu_freq': self.jetson.gpu.get('freq', 0),  # GPU frequency
                    'ram_used': self.jetson.memory.get('used', 0),  # RAM used (MB)
                    'ram_total': self.jetson.memory.get('total', 0),  # Total RAM (MB)
                    'temp': {
                        'gpu': self.jetson.temperature.get('GPU', 0),  # GPU temp
                        'cpu': self.jetson.temperature.get('CPU', 0),  # CPU temp
                    },
                    'power': self.jetson.power.get('tot', 0),  # Total power (mW)
                }
                self.stats.append(stats)
        except Exception as e:
            print(f"Error sampling Jetson stats: {e}")
    
    def stop(self):
        """Stop monitoring and return stats."""
        if not self.available or self.jetson is None:
            return {}
        
        try:
            self.jetson.close()
            
            # Calculate average values
            if len(self.stats) == 0:
                return {}
            
            avg_stats = {
                'gpu_util': np.mean([s['gpu'] for s in self.stats]),
                'gpu_freq': np.mean([s['gpu_freq'] for s in self.stats]),
                'ram_used': np.mean([s['ram_used'] for s in self.stats]),
                'ram_total': self.stats[0]['ram_total'],  # Total RAM shouldn't change
                'gpu_temp': np.mean([s['temp']['gpu'] for s in self.stats]),
                'cpu_temp': np.mean([s['temp']['cpu'] for s in self.stats]),
                'power_mw': np.mean([s['power'] for s in self.stats]),
                'samples': len(self.stats),
            }
            
            return avg_stats
        except Exception as e:
            print(f"Error stopping Jetson monitoring: {e}")
            return {}


def get_onnx_model_info(model_path: str, debug: bool = False) -> Dict[str, Any]:
    """Get basic information about an ONNX model."""
    info = {
        'path': model_path,
        'size_mb': os.path.getsize(model_path) / (1024 * 1024),
        'filename': os.path.basename(model_path),
    }
    
    try:
        # Get input and output names
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(model_path, sess_options)
        
        info['input_names'] = [inp.name for inp in session.get_inputs()]
        info['output_names'] = [out.name for out in session.get_outputs()]
        
        # Get input shapes
        info['input_shapes'] = {inp.name: inp.shape for inp in session.get_inputs()}
        info['input_types'] = {inp.name: inp.type for inp in session.get_inputs()}
        
        # Get output shapes
        info['output_shapes'] = {out.name: out.shape for out in session.get_outputs()}
        info['output_types'] = {out.name: out.type for out in session.get_outputs()}
        
        # Model opset version
        model = ort.InferenceSession(model_path).get_modelmeta()
        info['opset_version'] = model.custom_metadata_map.get('onnx_opset_version', 'unknown')
        
        if debug:
            print(f"Model: {model_path}")
            print(f"  Input names: {info['input_names']}")
            print(f"  Input shapes: {info['input_shapes']}")
            print(f"  Input types: {info['input_types']}")
            print(f"  Output names: {info['output_names']}")
            print(f"  Output shapes: {info['output_shapes']}")
            print(f"  Output types: {info['output_types']}")
            print(f"  Opset version: {info['opset_version']}")
        
        return info
    except Exception as e:
        print(f"Error getting ONNX model info for {model_path}: {e}")
        return info


def preprocess_image(image_path: str, target_width: int, target_height: int) -> np.ndarray:
    """Preprocess an image for model input."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Resize to target dimensions
    img_resized = cv2.resize(img, (target_width, target_height))
    
    # Convert to RGB (OpenCV loads as BGR)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    img_normalized = img_rgb.astype(np.float32) / 255.0
    
    # HWC to CHW format (Channel first for PyTorch/ONNX models)
    img_chw = np.transpose(img_normalized, (2, 0, 1))
    
    # Add batch dimension
    img_batch = np.expand_dims(img_chw, axis=0)
    
    return img_batch


def load_and_prepare_models(model_dir: str, use_cuda: bool, debug: bool = False) -> Dict[str, Dict[str, Any]]:
    """Load original and quantized encoder and decoder models."""
    models = {}
    
    # Define model paths
    encoder_path = os.path.join(model_dir, "encoder.onnx")
    decoder_path = os.path.join(model_dir, "depth.onnx")
    q_encoder_path = os.path.join(model_dir, "encoder_quantized.onnx")
    q_decoder_path = os.path.join(model_dir, "depth_quantized.onnx")
    
    # Check for model existence
    model_files = {
        "encoder": (encoder_path, os.path.exists(encoder_path)),
        "decoder": (decoder_path, os.path.exists(decoder_path)),
        "encoder_quantized": (q_encoder_path, os.path.exists(q_encoder_path)),
        "decoder_quantized": (q_decoder_path, os.path.exists(q_decoder_path))
    }
    
    # Report missing models
    missing = [name for name, (path, exists) in model_files.items() if not exists]
    if missing:
        print(f"Warning: The following models were not found: {', '.join(missing)}")
    
    # Set execution provider
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_cuda else ['CPUExecutionProvider']
    
    # Check if the requested provider is available
    available_providers = ort.get_available_providers()
    for provider in providers[:]:
        if provider not in available_providers:
            print(f"Warning: {provider} is not available. Available providers: {available_providers}")
            providers.remove(provider)
    
    if not providers:
        print("No valid execution providers available. Using default.")
        providers = ort.get_available_providers()
    
    print(f"Using execution providers: {providers}")
    
    # Create sessions for all available models
    for name, (path, exists) in model_files.items():
        if exists:
            try:
                # Get model info
                model_info = get_onnx_model_info(path, debug)
                
                # Create session
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                session = ort.InferenceSession(path, sess_options, providers=providers)
                
                models[name] = {
                    "session": session,
                    "info": model_info,
                    "providers": session.get_providers()
                }
                
                print(f"Loaded {name}: {path}")
                print(f"  Size: {model_info['size_mb']:.2f} MB")
                print(f"  Providers: {session.get_providers()}")
            except Exception as e:
                print(f"Error loading model {name} ({path}): {e}")
    
    return models


def run_inference(
    models: Dict[str, Dict[str, Any]],
    input_tensor: np.ndarray,
    num_warmup: int = 5,
    num_iter: int = 50,
    debug: bool = False
) -> Dict[str, Dict[str, Any]]:
    """Run inference benchmarks on original and quantized models."""
    results = {}
    
    # Check if we have both encoder and decoder models
    model_pairs = [
        ("original", ("encoder", "decoder")),
        ("quantized", ("encoder_quantized", "decoder_quantized"))
    ]
    
    for model_type, (encoder_key, decoder_key) in model_pairs:
        if encoder_key not in models or decoder_key not in models:
            print(f"Skipping {model_type} model benchmark (missing encoder or decoder)")
            continue
        
        print(f"\nBenchmarking {model_type} model...")
        encoder_session = models[encoder_key]["session"]
        decoder_session = models[decoder_key]["session"]
        
        # Get input/output names
        encoder_input_name = encoder_session.get_inputs()[0].name
        encoder_output_names = [output.name for output in encoder_session.get_outputs()]
        decoder_input_names = [inp.name for inp in decoder_session.get_inputs()]
        decoder_output_names = [output.name for output in decoder_session.get_outputs()]
        
        # Print detailed debug info about inputs and outputs
        print(f"\nModel {model_type} - Encoder/Decoder Interface:")
            # ...existing code...
        print(f"  Encoder outputs: {encoder_output_names}")
        print(f"  Decoder inputs: {decoder_input_names}")
        
        # Check encoder outputs and decoder inputs for compatibility
        print("\nAnalyzing model compatibility:")
        
        # Get shapes
        encoder_output_shapes = {out.name: out.shape for out in encoder_session.get_outputs()}
        decoder_input_shapes = {inp.name: inp.shape for inp in decoder_session.get_inputs()}
        
        print(f"  Encoder output shapes: {encoder_output_shapes}")
        print(f"  Decoder input shapes: {decoder_input_shapes}")
        
        # Try a test run to see the actual output
        print("\nRunning test inference to check actual output shapes...")
        test_input = input_tensor.copy()
        if model_type == "quantized":
            test_input = test_input.astype(np.float16)
        test_encoder_outputs = encoder_session.run(encoder_output_names, {encoder_input_name: test_input})
        
        print("  Actual encoder outputs:")
        for i, (name, output) in enumerate(zip(encoder_output_names, test_encoder_outputs)):
            print(f"    Output {i}: {name}, shape={output.shape}, dtype={output.dtype}")
        
        # Map encoder outputs to decoder inputs by shape
        print("\nMapping encoder outputs to decoder inputs by shape...")
        decoder_inputs = {}
        encoder_output_dict = {name: output for name, output in zip(encoder_output_names, test_encoder_outputs)}
        for decoder_input_name in decoder_input_names:
            expected_shape = tuple(decoder_input_shapes[decoder_input_name])
            found = False
            for name, arr in encoder_output_dict.items():
                if arr.shape == expected_shape:
                    decoder_inputs[decoder_input_name] = arr
                    print(f"  Mapped decoder input {decoder_input_name} <- encoder output {name} (shape={arr.shape})")
                    found = True
                    break
            if not found:
                print(f"  Warning: No encoder output matches shape for decoder input {decoder_input_name} (expected {expected_shape})")
        
        # Start Jetson monitoring
        jetson_monitor = JetsonMonitor()
        jetson_monitor.start()
        
        # Warmup
        print(f"Running {num_warmup} warmup iterations...")
        
        # Flag to track if warmup was successful
        warmup_success = True
        
        # Convert input tensor to float16 for quantized models
        model_input_tensor = input_tensor
        if model_type == "quantized":
            model_input_tensor = model_input_tensor.astype(np.float16)
        for _ in range(num_warmup):
            try:
                encoder_session.run(encoder_output_names, {encoder_input_name: model_input_tensor})
                decoder_session.run(decoder_output_names, decoder_inputs)
            except Exception as e:
                print(f"  Error during warmup: {e}")
                warmup_success = False
                break
        
        if not warmup_success:
            print(f"  Skipping {model_type} model benchmark due to warmup errors")
            continue
        
        # Benchmark
        print(f"Running {num_iter} benchmark iterations...")
        latencies = []
        
        for i in range(num_iter):
            jetson_monitor.sample()
            start_time = time.time()
            try:
                encoder_outputs = encoder_session.run(encoder_output_names, {encoder_input_name: model_input_tensor})
                encoder_output_dict = {name: output for name, output in zip(encoder_output_names, encoder_outputs)}
                # Remap decoder inputs for each iteration
                for decoder_input_name in decoder_input_names:
                    expected_shape = tuple(decoder_input_shapes[decoder_input_name])
                    found = False
                    for name, arr in encoder_output_dict.items():
                        if arr.shape == expected_shape:
                            decoder_inputs[decoder_input_name] = arr
                            found = True
                            break
                    if not found:
                        print(f"  Warning: No encoder output matches shape for decoder input {decoder_input_name} (expected {expected_shape})")
                decoder_outputs = decoder_session.run(decoder_output_names, decoder_inputs)
                end_time = time.time()
                latency = (end_time - start_time) * 1000  # ms
                latencies.append(latency)
                if (i + 1) % 10 == 0:
                    print(f"  Iteration {i+1}/{num_iter}: {latency:.2f} ms")
            except Exception as e:
                print(f"  Error during benchmark iteration {i+1}: {e}")
                continue
        
        # Get Jetson hardware stats
        hw_stats = jetson_monitor.stop()
        
        # Calculate metrics
        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        fps = 1000 / avg_latency
        
        # Get model sizes
        encoder_size = models[encoder_key]["info"]["size_mb"]
        decoder_size = models[decoder_key]["info"]["size_mb"]
        total_size = encoder_size + decoder_size
        
        results[model_type] = {
            "avg_latency_ms": float(avg_latency),
            "std_latency_ms": float(std_latency),
            "min_latency_ms": float(min_latency),
            "max_latency_ms": float(max_latency),
            "fps": float(fps),
            "model_size_mb": float(total_size),
            "encoder_size_mb": float(encoder_size),
            "decoder_size_mb": float(decoder_size),
            "num_iterations": num_iter,
            "providers": models[encoder_key]["providers"],
            "jetson_stats": hw_stats,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Print summary
        print(f"\n{model_type.capitalize()} Model Results:")
        print(f"  Average Latency: {avg_latency:.2f} ms (±{std_latency:.2f})")
        print(f"  FPS: {fps:.2f}")
        print(f"  Model Size: {total_size:.2f} MB (Encoder: {encoder_size:.2f} MB, Decoder: {decoder_size:.2f} MB)")
        
        if hw_stats:
            print(f"  GPU Utilization: {hw_stats['gpu_util']:.1f}%")
            print(f"  RAM Usage: {hw_stats['ram_used']:.1f} MB / {hw_stats['ram_total']:.1f} MB")
            print(f"  Power Consumption: {hw_stats['power_mw']:.1f} mW")
            print(f"  Temperature - GPU: {hw_stats['gpu_temp']:.1f}°C, CPU: {hw_stats['cpu_temp']:.1f}°C")
    
    return results


def write_results(results: Dict[str, Dict[str, Any]], output_dir: str) -> Dict[str, str]:
    """Write benchmark results to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_paths = {}
    
    # JSON output (all details)
    json_path = output_path / f"benchmark_results_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    file_paths["json"] = str(json_path)
    
    # CSV output (main metrics)
    csv_path = output_path / f"benchmark_results_{timestamp}.csv"
    fieldnames = [
        "model_type", "avg_latency_ms", "std_latency_ms", "min_latency_ms", "max_latency_ms", 
        "fps", "model_size_mb", "encoder_size_mb", "decoder_size_mb",
    ]
    
    # Add Jetson stats fields if available
    jetson_fields = []
    for model_type, model_results in results.items():
        if "jetson_stats" in model_results and model_results["jetson_stats"]:
            jetson_fields = list(model_results["jetson_stats"].keys())
            break
    
    if jetson_fields:
        fieldnames.extend([f"jetson_{field}" for field in jetson_fields])
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for model_type, model_results in results.items():
            row = {
                "model_type": model_type,
                "avg_latency_ms": model_results["avg_latency_ms"],
                "std_latency_ms": model_results["std_latency_ms"],
                "min_latency_ms": model_results["min_latency_ms"],
                "max_latency_ms": model_results["max_latency_ms"],
                "fps": model_results["fps"],
                "model_size_mb": model_results["model_size_mb"],
                "encoder_size_mb": model_results["encoder_size_mb"],
                "decoder_size_mb": model_results["decoder_size_mb"],
            }
            
            # Add Jetson stats if available
            if "jetson_stats" in model_results and model_results["jetson_stats"]:
                for field in jetson_fields:
                    row[f"jetson_{field}"] = model_results["jetson_stats"].get(field, "")
            
            writer.writerow(row)
    
    file_paths["csv"] = str(csv_path)
    
    # Markdown summary
    md_path = output_path / f"benchmark_results_{timestamp}.md"
    with open(md_path, 'w') as f:
        f.write("# Monodepth2 ONNX Model Benchmark Results\n\n")
        f.write(f"Benchmark run on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Model Performance Comparison\n\n")
        f.write("| Model Type | FPS | Latency (ms) | Model Size (MB) |\n")
        f.write("|------------|-----|--------------|----------------|\n")
        
        for model_type, model_results in results.items():
            f.write(
                f"| {model_type.capitalize()} | "
                f"{model_results['fps']:.2f} | "
                f"{model_results['avg_latency_ms']:.2f} ± {model_results['std_latency_ms']:.2f} | "
                f"{model_results['model_size_mb']:.2f} |\n"
            )
        
        f.write("\n## Jetson Hardware Metrics\n\n")
        
        has_jetson_stats = any("jetson_stats" in model_results and model_results["jetson_stats"] 
                               for model_results in results.values())
        
        if has_jetson_stats:
            f.write("| Model Type | GPU Utilization (%) | RAM Usage (MB) | Power (mW) | GPU Temp (°C) | CPU Temp (°C) |\n")
            f.write("|------------|---------------------|----------------|------------|---------------|---------------|\n")
            
            for model_type, model_results in results.items():
                if "jetson_stats" in model_results and model_results["jetson_stats"]:
                    stats = model_results["jetson_stats"]
                    f.write(
                        f"| {model_type.capitalize()} | "
                        f"{stats.get('gpu_util', 'N/A'):.1f} | "
                        f"{stats.get('ram_used', 'N/A'):.1f} | "
                        f"{stats.get('power_mw', 'N/A'):.1f} | "
                        f"{stats.get('gpu_temp', 'N/A'):.1f} | "
                        f"{stats.get('cpu_temp', 'N/A'):.1f} |\n"
                    )
        else:
            f.write("No Jetson hardware metrics available.\n")
    
    file_paths["md"] = str(md_path)
    
    return file_paths


def plot_results(results: Dict[str, Dict[str, Any]], output_dir: str) -> Dict[str, str]:
    """Create visualizations of benchmark results."""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib is not available. Skipping plot generation.")
        return {}
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_paths = {}
    
    # Check if we have both model types
    if len(results) < 2:
        print("Need both original and quantized models for comparison plots.")
        return plot_paths
    
    # Extract model types
    model_types = list(results.keys())
    
    # 1. Performance comparison plot (FPS and Latency)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # FPS comparison
    fps_values = [results[model_type]["fps"] for model_type in model_types]
    ax1.bar(model_types, fps_values, color=['tab:blue', 'tab:orange'])
    ax1.set_title('FPS Comparison')
    ax1.set_ylabel('Frames Per Second')
    ax1.grid(True, alpha=0.3)
    
    # Add percentage improvement/degradation
    if len(model_types) == 2:
        fps_change = ((results[model_types[1]]["fps"] - results[model_types[0]]["fps"]) / 
                       results[model_types[0]]["fps"] * 100)
        change_label = f"{fps_change:.1f}% {'improvement' if fps_change > 0 else 'degradation'}"
        ax1.text(0.5, 0.9, change_label, transform=ax1.transAxes, ha='center',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    
    # Latency comparison
    latency_values = [results[model_type]["avg_latency_ms"] for model_type in model_types]
    latency_std = [results[model_type]["std_latency_ms"] for model_type in model_types]
    ax2.bar(model_types, latency_values, yerr=latency_std, color=['tab:blue', 'tab:orange'])
    ax2.set_title('Inference Latency Comparison')
    ax2.set_ylabel('Latency (ms)')
    ax2.grid(True, alpha=0.3)
    
    # Add percentage improvement/degradation
    if len(model_types) == 2:
        latency_change = ((results[model_types[0]]["avg_latency_ms"] - results[model_types[1]]["avg_latency_ms"]) / 
                           results[model_types[0]]["avg_latency_ms"] * 100)
        change_label = f"{latency_change:.1f}% {'improvement' if latency_change > 0 else 'degradation'}"
        ax2.text(0.5, 0.9, change_label, transform=ax2.transAxes, ha='center',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    
    fig.tight_layout()
    perf_plot_path = output_path / f"performance_comparison_{timestamp}.png"
    fig.savefig(perf_plot_path, dpi=150)
    plt.close(fig)
    plot_paths["performance"] = str(perf_plot_path)
    
    # 2. Jetson RAM usage comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    ram_used = []
    for model_type in model_types:
        stats = results[model_type].get("jetson_stats", {})
        ram_used.append(stats.get("ram_used", float('nan')))
    x = np.arange(len(model_types))
    width = 0.35
    ax.bar(x, ram_used, width, color=['tab:blue', 'tab:orange'])
    ax.set_ylabel('RAM Usage (MB)')
    ax.set_title('Jetson RAM Usage Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in model_types])
    ax.grid(True, alpha=0.3)
    if len(model_types) == 2 and all(not np.isnan(val) for val in ram_used):
        ram_change = ((ram_used[1] - ram_used[0]) / ram_used[0] * 100) if ram_used[0] else 0
        change_label = f"{ram_change:+.1f}% RAM change"
        ax.text(0.5, 0.9, change_label, transform=ax.transAxes, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    ram_plot_path = output_path / f"jetson_ram_comparison_{timestamp}.png"
    fig.tight_layout()
    fig.savefig(ram_plot_path, dpi=150)
    plt.close(fig)
    plot_paths["ram"] = str(ram_plot_path)
    
    # 3. Jetson hardware metrics comparison (if available)
    has_jetson_stats = all("jetson_stats" in model_results and model_results["jetson_stats"] 
                          for model_results in results.values())
    
    if has_jetson_stats:
        # Plot GPU utilization, RAM usage, and power
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # GPU utilization
        gpu_util = [results[model_type]["jetson_stats"]["gpu_util"] for model_type in model_types]
        ax1.bar(model_types, gpu_util, color=['tab:blue', 'tab:orange'])
        ax1.set_title('GPU Utilization')
        ax1.set_ylabel('Utilization (%)')
        ax1.grid(True, alpha=0.3)
        
        # RAM usage
        ram_used = [results[model_type]["jetson_stats"]["ram_used"] for model_type in model_types]
        ax2.bar(model_types, ram_used, color=['tab:blue', 'tab:orange'])
        ax2.set_title('RAM Usage')
        ax2.set_ylabel('Usage (MB)')
        ax2.grid(True, alpha=0.3)
        
        # Power consumption
        power = [results[model_type]["jetson_stats"]["power_mw"] for model_type in model_types]
        ax3.bar(model_types, power, color=['tab:blue', 'tab:orange'])
        ax3.set_title('Power Consumption')
        ax3.set_ylabel('Power (mW)')
        ax3.grid(True, alpha=0.3)
        
        fig.tight_layout()
        hw_plot_path = output_path / f"jetson_hardware_metrics_{timestamp}.png"
        fig.savefig(hw_plot_path, dpi=150)
        plt.close(fig)
        plot_paths["hardware"] = str(hw_plot_path)
    
    # 4. Combined summary plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    # FPS
    ax1.bar(model_types, fps_values, color=['tab:blue', 'tab:orange'])
    ax1.set_title('FPS')
    ax1.set_ylabel('Frames Per Second')
    ax1.grid(True, alpha=0.3)
    # Latency
    ax2.bar(model_types, latency_values, yerr=latency_std, color=['tab:blue', 'tab:orange'])
    ax2.set_title('Latency (ms)')
    ax2.set_ylabel('Milliseconds')
    ax2.grid(True, alpha=0.3)
    # Jetson RAM Usage
    ax3.bar(model_types, ram_used, color=['tab:blue', 'tab:orange'])
    ax3.set_title('Jetson RAM Usage (MB)')
    ax3.set_ylabel('Megabytes')
    ax3.grid(True, alpha=0.3)
    # GPU Utilization
    if has_jetson_stats:
        gpu_util = [results[model_type]["jetson_stats"].get("gpu_util", float('nan')) for model_type in model_types]
        if all(not np.isnan(val) for val in gpu_util):
            ax4.bar(model_types, gpu_util, color=['tab:blue', 'tab:orange'])
            ax4.set_title('GPU Utilization (%)')
            ax4.set_ylabel('Percentage')
        else:
            ax4.text(0.5, 0.5, 'Jetson GPU stats not available', 
                     ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('GPU Utilization')
    else:
        ax4.text(0.5, 0.5, 'Jetson stats not available', 
                 ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('GPU Utilization')
    ax4.grid(True, alpha=0.3)
    fig.suptitle('Monodepth2 ONNX Models Benchmark Summary', fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.94)
    summary_plot_path = output_path / f"benchmark_summary_{timestamp}.png"
    fig.savefig(summary_plot_path, dpi=150)
    plt.close(fig)
    plot_paths["summary"] = str(summary_plot_path)
    
    return plot_paths


def run_comparison_inference(input_tensor, encoder, decoder, num_iter=10):
    """Run inference on a single input with both encoder and decoder."""
    encoder_input_name = encoder.get_inputs()[0].name
    encoder_output_names = [output.name for output in encoder.get_outputs()]
    decoder_input_names = [inp.name for inp in decoder.get_inputs()]
    decoder_output_names = [output.name for output in decoder.get_outputs()]
    
    # Convert input to float16 for quantized model
    if any('quant' in p.lower() for p in encoder.get_providers()):
        input_tensor = input_tensor.astype(np.float16)
    encoder_outputs = encoder.run(encoder_output_names, {encoder_input_name: input_tensor})
    # Map encoder outputs to decoder inputs by shape
    decoder_inputs = {}
    encoder_output_dict = {name: output for name, output in zip(encoder_output_names, encoder_outputs)}
    for decoder_input_name in decoder_input_names:
        expected_shape = decoder.get_inputs()[decoder_input_names.index(decoder_input_name)].shape
        found = False
        for name, arr in encoder_output_dict.items():
            if tuple(arr.shape) == tuple(expected_shape):
                decoder_inputs[decoder_input_name] = arr
                found = True
                break
        if not found:
            print(f"Warning: No encoder output matches shape for decoder input {decoder_input_name} (expected {expected_shape})")
    try:
        decoder_outputs = decoder.run(decoder_output_names, decoder_inputs)
        return decoder_outputs
    except Exception as e:
        print(f"Error running decoder in comparison inference: {e}")
        return None


def generate_depth_visualization(results, input_image_path, output_dir, width=640, height=192):
    """Generate depth visualization for original and quantized models."""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib is not available. Skipping depth visualization.")
        return {}
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    vis_paths = {}
    
    # Prepare input image
    input_tensor = preprocess_image(input_image_path, width, height)
    
    # Load original image for visualization
    original_img = cv2.imread(input_image_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    original_img = cv2.resize(original_img, (width, height))
    
    # Run inference with each model pair and visualize
    for model_type in ["original", "quantized"]:
        encoder_key = "encoder" if model_type == "original" else "encoder_quantized"
        decoder_key = "decoder" if model_type == "original" else "decoder_quantized"
        
        if encoder_key not in models or decoder_key not in models:
            continue
        
        encoder_session = models[encoder_key]["session"]
        decoder_session = models[decoder_key]["session"]
        
        # Run inference
        try:
            decoder_outputs = run_comparison_inference(input_tensor, encoder_session, decoder_session)
            if decoder_outputs is None:
                print(f"Skipping visualization for {model_type} model due to inference errors")
                continue
                
            # Extract disparity (assume first output is disparity)
            disp = decoder_outputs[0]
            
            # Normalize for visualization
            disp_resized = cv2.resize(disp[0, 0], (width, height))
            disp_norm = (disp_resized - disp_resized.min()) / (disp_resized.max() - disp_resized.min())
        except Exception as e:
            print(f"Error during visualization for {model_type} model: {e}")
            continue
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.imshow(original_img)
        ax1.set_title('Input Image')
        ax1.axis('off')
        
        depth_viz = ax2.imshow(disp_norm, cmap='magma')
        ax2.set_title(f'Predicted Depth ({model_type.capitalize()})')
        ax2.axis('off')
        
        fig.colorbar(depth_viz, ax=ax2, fraction=0.046, pad=0.04)
        fig.suptitle(f'Depth Prediction - {model_type.capitalize()} Model')
        
        vis_path = output_path / f"depth_visualization_{model_type}_{timestamp}.png"
        fig.tight_layout()
        fig.savefig(vis_path, dpi=150)
        plt.close(fig)
        
        vis_paths[model_type] = str(vis_path)
    
    # Create side-by-side comparison if both models are available
    if len(vis_paths) == 2:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        ax1.imshow(original_img)
        ax1.set_title('Input Image')
        ax1.axis('off')
        
        # Get the depth maps - handle potential errors
        try:
            orig_outputs = run_comparison_inference(
                input_tensor, 
                models["encoder"]["session"],
                models["decoder"]["session"]
            )
            if orig_outputs is None:
                print("Error generating original model depth map")
                return vis_paths
            
            quant_outputs = run_comparison_inference(
                input_tensor, 
                models["encoder_quantized"]["session"],
                models["decoder_quantized"]["session"]
            )
            if quant_outputs is None:
                print("Error generating quantized model depth map")
                return vis_paths
                
            orig_disp = orig_outputs[0]
            quant_disp = quant_outputs[0]
            
            # Normalize for visualization
            orig_disp_resized = cv2.resize(orig_disp[0, 0], (width, height))
            orig_disp_norm = (orig_disp_resized - orig_disp_resized.min()) / (orig_disp_resized.max() - orig_disp_resized.min())
            
            quant_disp_resized = cv2.resize(quant_disp[0, 0], (width, height))
            quant_disp_norm = (quant_disp_resized - quant_disp_resized.min()) / (quant_disp_resized.max() - quant_disp_resized.min())
        except Exception as e:
            print(f"Error during comparison visualization: {e}")
            return vis_paths
        
        # Plot original depth
        depth_viz1 = ax2.imshow(orig_disp_norm, cmap='magma')
        ax2.set_title('Original Model')
        ax2.axis('off')
        
        # Plot quantized depth
        depth_viz2 = ax3.imshow(quant_disp_norm, cmap='magma')
        ax3.set_title('Quantized Model')
        ax3.axis('off')
        
        fig.colorbar(depth_viz1, ax=ax2, fraction=0.046, pad=0.04)
        fig.colorbar(depth_viz2, ax=ax3, fraction=0.046, pad=0.04)
        
        fig.suptitle('Depth Prediction Comparison: Original vs Quantized')
        
        compare_path = output_path / f"depth_comparison_{timestamp}.png"
        fig.tight_layout()
        fig.savefig(compare_path, dpi=150)
        plt.close(fig)
        
        vis_paths["comparison"] = str(compare_path)
        
        # Calculate and visualize the difference between original and quantized
        diff = np.abs(orig_disp_norm - quant_disp_norm)
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        ax1.imshow(orig_disp_norm, cmap='magma')
        ax1.set_title('Original Model')
        ax1.axis('off')
        
        ax2.imshow(quant_disp_norm, cmap='magma')
        ax2.set_title('Quantized Model')
        ax2.axis('off')
        
        diff_viz = ax3.imshow(diff, cmap='viridis')
        ax3.set_title('Absolute Difference')
        ax3.axis('off')
        
        fig.colorbar(diff_viz, ax=ax3, fraction=0.046, pad=0.04)
        
        fig.suptitle('Depth Prediction Difference: Original vs Quantized')
        
        diff_path = output_path / f"depth_difference_{timestamp}.png"
        fig.tight_layout()
        fig.savefig(diff_path, dpi=150)
        plt.close(fig)
        
        vis_paths["difference"] = str(diff_path)
    
    return vis_paths


def main():
    """Main benchmark function."""
    args = parse_args()
    
    print(f"ONNX Runtime version: {ort.__version__}")
    print(f"Available providers: {ort.get_available_providers()}")
    print(f"Model path: {args.model_path}")
    print(f"Image path: {args.image_path}")
    print(f"Target dimensions: {args.width}x{args.height}")
    print(f"Batch size: {args.batch_size}")
    print(f"Use CUDA: {args.use_cuda}")
    print(f"Debug mode: {args.debug}")
    
    # Load models
    global models
    models = load_and_prepare_models(args.model_path, args.use_cuda, args.debug)
    
    if not models:
        print("No models found. Exiting.")
        return
    
    # Prepare input image
    input_tensor = preprocess_image(args.image_path, args.width, args.height)
    
    # If batch size > 1, duplicate the input tensor
    if args.batch_size > 1:
        input_tensor = np.repeat(input_tensor, args.batch_size, axis=0)
        print(f"Created batch of size {args.batch_size} with shape {input_tensor.shape}")
    
    # Run benchmarks
    results = run_inference(
        models,
        input_tensor,
        num_warmup=args.num_warmup,
        num_iter=args.num_iter,
        debug=args.debug
    )
    
    if not results:
        print("Benchmark failed. No results available.")
        return
    
    # Write results to files
    file_paths = write_results(results, args.results_dir)
    print("\nResults saved to:")
    for file_type, path in file_paths.items():
        print(f"  {file_type}: {path}")
    
    # Generate plots
    if MATPLOTLIB_AVAILABLE:
        plot_paths = plot_results(results, args.results_dir)
        print("\nPlots saved to:")
        for plot_type, path in plot_paths.items():
            print(f"  {plot_type}: {path}")
        
        # Generate depth visualization
        vis_paths = generate_depth_visualization(
            results,
            args.image_path,
            args.results_dir,
            args.width,
            args.height
        )
        print("\nDepth visualizations saved to:")
        for vis_type, path in vis_paths.items():
            print(f"  {vis_type}: {path}")
    
    # Print comparison summary if we have both model types
    if len(results) >= 2:
        print("\nPerformance Comparison Summary:")
        orig_fps = results["original"]["fps"]
        quant_fps = results["quantized"]["fps"]
        fps_change = ((quant_fps - orig_fps) / orig_fps * 100)
        
        orig_latency = results["original"]["avg_latency_ms"]
        quant_latency = results["quantized"]["avg_latency_ms"]
        latency_change = ((orig_latency - quant_latency) / orig_latency * 100)
        
        orig_size = results["original"]["model_size_mb"]
        quant_size = results["quantized"]["model_size_mb"]
        size_change = ((orig_size - quant_size) / orig_size * 100)
        
        print(f"  FPS: {orig_fps:.2f} → {quant_fps:.2f} ({fps_change:+.1f}%)")
        print(f"  Latency: {orig_latency:.2f}ms → {quant_latency:.2f}ms ({latency_change:+.1f}%)")
        print(f"  Model Size: {orig_size:.2f}MB → {quant_size:.2f}MB ({size_change:+.1f}%)")
        
        # Print Jetson stats comparison if available
        has_jetson_stats = all("jetson_stats" in model_results and model_results["jetson_stats"] 
                              for model_results in results.values())
        
        if has_jetson_stats:
            print("\nJetson Hardware Metrics Comparison:")
            orig_gpu = results["original"]["jetson_stats"]["gpu_util"]
            quant_gpu = results["quantized"]["jetson_stats"]["gpu_util"]
            gpu_change = ((quant_gpu - orig_gpu) / orig_gpu * 100) if orig_gpu else 0
            
            orig_ram = results["original"]["jetson_stats"]["ram_used"]
            quant_ram = results["quantized"]["jetson_stats"]["ram_used"]
            ram_change = ((quant_ram - orig_ram) / orig_ram * 100) if orig_ram else 0
            
            orig_power = results["original"]["jetson_stats"]["power_mw"]
            quant_power = results["quantized"]["jetson_stats"]["power_mw"]
            power_change = ((quant_power - orig_power) / orig_power * 100) if orig_power else 0
            
            print(f"  GPU Utilization: {orig_gpu:.1f}% → {quant_gpu:.1f}% ({gpu_change:+.1f}%)")
            print(f"  RAM Usage: {orig_ram:.1f}MB → {quant_ram:.1f}MB ({ram_change:+.1f}%)")
            print(f"  Power Consumption: {orig_power:.1f}mW → {quant_power:.1f}mW ({power_change:+.1f}%)")
    
    print("\nBenchmark completed successfully!")


if __name__ == "__main__":
    main()
