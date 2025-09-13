# Jetson TX2 ONNX Inference Benchmark

This tool benchmarks the original and quantized ONNX models on NVIDIA Jetson TX2, measuring performance metrics like FPS, latency, GPU usage, and memory usage. It generates comprehensive comparisons and visualizations of the results.

## Features

- Benchmarks both original and quantized ONNX models
- Measures inference speed (FPS and latency)
- Monitors Jetson-specific hardware metrics (GPU usage, memory usage, power consumption, temperature)
- Generates detailed comparison visualizations
- Creates depth prediction visualizations to compare model output quality
- Exports results in multiple formats (JSON, CSV, Markdown)

## Requirements

- Jetson TX2 with JetPack (tested on JetPack 4.4+)
- Python 3.6
- ONNX Runtime for Jetson
- Other dependencies listed in `requirements_jetson.txt`

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/monodepth2_forked.git
cd monodepth2_forked
```

2. Install dependencies:

```bash
pip3 install -r requirements_jetson.txt
```

3. Install ONNX Runtime:

For Jetson, you may need to build ONNX Runtime from source or use a pre-built wheel that is compatible with Jetson TX2. Follow NVIDIA's guidelines for installing ONNX Runtime on Jetson devices.

Example (using pre-built wheel):

```bash
pip3 install onnxruntime-gpu==1.5.2
```

## Usage

Basic usage:

```bash
python3 jetson_benchmark.py
```

This will use the default settings:

- Model path: ./models/mono+stereo_640x192
- Test image: ./assets/test_image.jpg
- Input dimensions: 640x192
- 5 warmup iterations, 50 benchmark iterations

### Command Line Options

```
usage: jetson_benchmark.py [-h] [--model_path MODEL_PATH] [--image_path IMAGE_PATH]
                          [--width WIDTH] [--height HEIGHT] [--num_warmup NUM_WARMUP]
                          [--num_iter NUM_ITER] [--results_dir RESULTS_DIR]
                          [--batch_size BATCH_SIZE] [--use_cuda]

Benchmark ONNX models on Jetson Nano TX2

optional arguments:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        Path to the directory containing ONNX models
  --image_path IMAGE_PATH
                        Path to test image or directory of images
  --width WIDTH         Input image width
  --height HEIGHT       Input image height
  --num_warmup NUM_WARMUP
                        Number of warmup iterations
  --num_iter NUM_ITER   Number of benchmark iterations
  --results_dir RESULTS_DIR
                        Directory to save benchmark results
  --batch_size BATCH_SIZE
                        Batch size for inference
  --use_cuda            Use CUDA execution provider if available
```

### Examples

1. Basic benchmark with CUDA acceleration:

```bash
python3 jetson_benchmark.py --use_cuda
```

2. Custom model path and test image:

```bash
python3 jetson_benchmark.py --model_path /path/to/models --image_path /path/to/test_image.jpg
```

3. Custom image dimensions:

```bash
python3 jetson_benchmark.py --width 1280 --height 384
```

4. More benchmark iterations:

```bash
python3 jetson_benchmark.py --num_warmup 10 --num_iter 100
```

## Output

The script generates the following outputs in the results directory:

1. Numerical results:

   - `benchmark_results_TIMESTAMP.json`: Detailed benchmark results in JSON format
   - `benchmark_results_TIMESTAMP.csv`: Summary of benchmark results in CSV format
   - `benchmark_results_TIMESTAMP.md`: Formatted Markdown summary

2. Performance comparison charts:

   - `performance_comparison_TIMESTAMP.png`: FPS and latency comparison
   - `model_size_comparison_TIMESTAMP.png`: Model size comparison
   - `jetson_hardware_metrics_TIMESTAMP.png`: Jetson hardware metrics comparison
   - `benchmark_summary_TIMESTAMP.png`: Combined summary of all metrics

3. Depth prediction visualizations:
   - `depth_visualization_original_TIMESTAMP.png`: Original model depth prediction
   - `depth_visualization_quantized_TIMESTAMP.png`: Quantized model depth prediction
   - `depth_comparison_TIMESTAMP.png`: Side-by-side comparison of depth predictions
   - `depth_difference_TIMESTAMP.png`: Visualization of differences between original and quantized predictions

## Troubleshooting

1. **ONNX Runtime issues**: Make sure you have the appropriate version of ONNX Runtime for your Jetson device. You may need to build it from source for optimal performance.

2. **ImportError for jetson-stats**: Install with `sudo pip3 install jetson-stats` and reboot your device.

3. **Permission issues**: If you encounter permission issues with Jetson hardware monitoring, run the script with sudo:

```bash
sudo python3 jetson_benchmark.py
```

4. **Memory errors**: Reduce batch size if you encounter out-of-memory errors.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
