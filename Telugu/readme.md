# OCR Model Docker Setup

This repository contains a Docker-based OCR (Optical Character Recognition) model that can process images and extract text in multiple languages.

## Prerequisites

- Docker installed on your system
- NVIDIA Docker runtime (for GPU support)
- NVIDIA drivers compatible with CUDA
- Sufficient disk space for model files and data

## Directory Structure

Organize your files as follows:

```
project/
├── Dockerfile
├── requirements.txt
├──[your OCR model code]
├──checkpoints/
│       └── [your_model_checkpoint.ckpt]
└── data/
    ├── input/
    │   └── [input images]
    └── output/
        └── [processed results will appear here]
```

## Setup Instructions

### 1. Build the Docker Image

```bash
sudo docker build -t bhaashaocr .
```

### 2. Prepare Your Data

- Place your input images in the `data/input/` directory
- Ensure your model checkpoint file is in the `model/checkpoints/` directory
- The `data/output/` directory will be created automatically for results

### 3. Run the OCR Model

```bash
docker run --rm --gpus all \
    --name bhaashaocr-container \
    -v /path/to/your/model:/model:ro \
    -v /path/to/your/data:/data \
    bhaashaocr \
    --pretrained /model/checkpoints/your_model.ckpt \
    --input_dir /data/input/ \
    --out_dir /data/output/
```

### Command Explanation

- `--rm`: Automatically remove container when it exits
- `--gpus all`: Enable GPU support for all available GPUs
- `--name bhaashaocr-container`: Assign a name to the running container
- `-v /path/to/your/model:/model:ro`: Mount model directory as read-only
- `-v /path/to/your/data:/data`: Mount data directory for input/output
- `--pretrained`: Path to the model checkpoint file
- `--input_dir`: Directory containing input images
- `--out_dir`: Directory where processed results will be saved


### Supported File Formats

- Input: JPEG, PNG
- Output: TXT


## Output Format

The OCR results will be saved in the specified output directory with:
- Extracted text files
- Confidence scores


**Note:** Replace `/path/to/your/model` and `/path/to/your/data` with your actual directory paths when running the commands.