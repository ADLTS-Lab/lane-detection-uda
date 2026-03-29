# Lane Detection with U-Net and Unsupervised Domain Adaptation (UDA)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/)

This is a lane detection system using a U‑Net with a ResNet34 encoder, trained on the **MoLane dataset** (part of the CARLANE benchmark). It combines supervised learning on the source domain with **unsupervised domain adaptation (UDA)** using entropy minimization to improve generalization on a target domain. The project was developed as part of a BSc capstone project **(ADLTS)**.

## Features

- **Supervised training** on labeled source data.
- **Unsupervised domain adaptation** with entropy minimization on unlabeled target data.
- **Comprehensive evaluation**: mIoU, F1-score, Brier score, pixel entropy, false positive/negative rates.
- **Visualization**: side‑by‑side comparisons of images, ground truth, and predictions.
- **Model export**: final model saved in `.pth`, `ONNX`, and `TorchScript` formats.
- **HTML report**: automatically generated performance summary with embedded images.
- **Fully configurable** via YAML files.
- **Modular codebase** – easy to extend or adapt to other datasets.

## Table of Contents

- [Getting Started](#getting-started)
- [Dataset](#dataset)
- [Configuration](#configuration)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [License](#license)

## Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)

### Installation

#### 1. Clone the repository:
   ```bash
   git clone https://github.com/ADLTS-Lab/lane-detection-uda.git
   cd lane-detection-uda
   ```
#### 2. Create a virtual environment (optional):
```bash
python -m venv venv
source venv/bin/activate  
```
On Windows: ```bash venv\Scripts\activate```

#### 3. Install dependencies:
```bash
pip install -r requirements.txt
```
#### 4. Download the MoLane dataset (see Dataset).

## Dataset
This project uses the MoLane subset of the **CARLANE benchmark**. The dataset is distributed under the Apache 2.0 license.

### Structure
After downloading, place the data in the data/ directory with the following structure:
```text
data/
└── carlane-benchmark/
    └── CARLANE/
        └── MoLane/
            ├── data/
            │   ├── 0000_image.jpg
            │   ├── 0000_label.png
            │   └── ...
            └── splits/
                ├── source_train.txt
                ├── source_val.txt
                ├── target_train.txt
                ├── target_val.txt
                └── target_test.txt
```
The split files contain relative paths to the images (and masks for labeled splits). The format is either:
- image_path mask_path (labeled)
- image_path (unlabeled)

Update the paths in configs/default.yaml accordingly.

## Configuration
All hyperparameters and paths are managed via YAML configuration files. The default file is `configs/default.yaml`:
```yaml
data:
  root: /path/to/carlane/MoLane/data
  splits_root: /path/to/carlane/MoLane/splits
  img_height: 360
  img_width: 640

training:
  batch_size: 8
  num_epochs: 50
  learning_rate: 1e-4
  device: cuda
  seed: 42

uda:
  alpha: 0.1
  learning_rate: 5e-5

model:
  n_classes: 2
  pretrained_weights: true

logging:
  log_dir: results/logs
  save_interval: 5

output_dir: .
```

## Usage
The entire pipeline is orchestrated by `run.py`. To run the complete pipeline (supervised training → adaptation → evaluation → conversion → report), use:
```bash
python run.py --config configs/default.yaml --mode all
```

For incremental steps:
| Mode	| Description |
| --- | --- |
| supervised | Train only the supervised model. |
| adaptation	| Run UDA (requires supervised model). |
| evaluate	| Evaluate both models and select the final one. |
| convert	| Export final model to ONNX and TorchScript. |
| report	| Generate HTML report from saved metrics. |

Example:
```bash
python run.py --config configs/hpc.yaml --mode supervised
```
### Inference on a single image
After training, you can run inference on a single image using `scripts/inference.py`:
```bash
python scripts/inference.py --model models/pretrained/final_lane_model.pth --image /path/to/image.jpg --output output.png
```
## Results
After running the full pipeline, you will find:

- Models in `models/pretrained/`:

   - `best_source_model.pth`

   - `best_adapted_model.pth`

   - `final_lane_model.pth`

   - `final_lane_model.onnx` (ONNX)

   - `final_lane_model.pt` (TorchScript)

- **Metrics** in `results/metrics/` as JSON files.

- **Visualizations** in `results/visualizations/` (prediction overlays).

- **HTML report** in `results/reports/evaluation_report.html`.

A sample report (as shown above) includes a summary table with **mIoU**, **F1**, **Brier score**, **pixel entropy**, and **FP/FN rates**, along with sample predictions.

## Project Structure
```text
lane-detection-uda/
├── run.py
├── configs/
├── data/                # Dataset
├── models/pretrained/
├── results/             # Logs, metrics, visualizations, reports
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/ 
│   └── utils/
├── scripts/                
├── notebooks/            # Exploration notebooks
└── tests/                # Unit tests
```
## Citation
If you use this code in your research, please cite:
```bibtex
@misc{lane-detection-uda,
  author = {Aelaf Tsegaye Getaneh},
  title = {Lane Detection with U-Net and Unsupervised Domain Adaptation},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ADLTS-Lab/lane-detection-uda}}
}
```
Additionally, please acknowledge the CARLANE benchmark:
```text
@inproceedings{carlane2020,
  title={CARLANE: A Lane Detection Benchmark for Urban Driving},
  author={G. B. and others},
  booktitle={CVPR Workshops},
  year={2020}
}
```
## License
This project is licensed under the Apache 2.0 License – see the [LICENSE](./LICENSE) file for details.