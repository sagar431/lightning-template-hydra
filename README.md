# PyTorch Lightning Template with Hydra

A template for PyTorch Lightning projects with planned Hydra integration, currently showcasing a high-accuracy dog breed classifier.

## Project Status

✅ **Current Implementation**: 
- Dog breed classifier using PyTorch Lightning
- Modern FastAPI web interface
- Docker and DevContainer support
- GPU acceleration with CUDA

🚧 **In Progress**: 
- Hydra integration for configuration management
- Experiment tracking
- Hyperparameter optimization
- Multi-run support

## Dataset

This project uses a dog breed classification dataset from Google Drive. The data should be organized in the following structure:

```
data/
└── dog_breed/
    └── dataset/
        ├── train/
        │   ├── beagle/
        │   ├── boxer/
        │   ├── bulldog/
        │   └── ...
        └── val/
            ├── beagle/
            ├── boxer/
            ├── bulldog/
            └── ...
```

Each breed directory contains the corresponding dog images. The dataset will be automatically downloaded when you first run the training script.

## Web Interface Demo

![Web Interface Demo](./assets/web_interface.png)

*Example: The model correctly identifies a Boxer with high confidence*

## Model Performance

The model achieves exceptional accuracy across supported breeds:

```
Recent Inference Results:
✓ Labrador Retriever (99.50% confidence)
✓ Boxer             (99.99% confidence)
✓ German Shepherd   (99.86% confidence)
✓ Bulldog           (98.05% confidence)
✓ Rottweiler        (99.50% confidence)
Average Accuracy: 100%
```

## Technical Stack

### Core Components
- **Framework**: PyTorch Lightning
- **Model**: ResNet18 from `timm` with pretrained weights
- **Web Interface**: FastAPI with modern UI
- **Package Manager**: `uv` for fast, reliable dependency management

### Training Configuration
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam
- **Batch Size**: 32
- **Epochs**: 10
- **Augmentation**: Random flips and rotations
- **Checkpointing**: Best models saved based on validation accuracy

### Development Tools
- **Container**: Docker with CUDA support
- **IDE Support**: VS Code DevContainer
- **GPU Support**: NVIDIA Container Toolkit
- **Dependencies**: Managed with `uv`

## Quick Start

### Using Docker (Recommended)
```bash
# Clone repository
git clone <your-repo-url>
cd lightning-template-hydra

# Start with Docker Compose
docker compose up --build

# Open browser at http://localhost:8080
```

### Manual Setup
```bash
# Clone repository
git clone <your-repo-url>
cd lightning-template-hydra

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .

# Start web interface
cd src
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

## Development with VS Code

This project includes a complete DevContainer configuration providing:
- Python extensions and tools
- NVIDIA GPU support
- Volume mounts for data persistence
- Fast package installation with `uv`
- Hot reload for development

To use:
1. Install [VS Code](https://code.visualstudio.com/) and [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
2. Open project in VS Code
3. Click "Reopen in Container"

## Command Line Interface

### Training
```bash
python src/train.py
```

### Evaluation
```bash
python src/eval.py --checkpoint_path checkpoints/last.ckpt
```

### Inference
```bash
python src/infer.py --checkpoint_path checkpoints/last.ckpt --num_images 9
```

## Supported Dog Breeds

1. Beagle
2. Boxer
3. Bulldog
4. Chihuahua
5. Dachshund
6. German Shepherd
7. Golden Retriever
8. Labrador Retriever
9. Poodle
10. Rottweiler

## Project Structure
```
.
├── checkpoints/          # Model checkpoints
├── data/                # Dataset directory
├── src/
│   ├── app.py          # FastAPI web application
│   ├── train.py        # Training script
│   ├── eval.py         # Evaluation script
│   ├── infer.py        # Inference script
│   ├── templates/      # HTML templates
│   └── static/         # Static files
├── Dockerfile          # Docker configuration
├── docker-compose.yml  # Docker Compose configuration
└── pyproject.toml      # Python dependencies
```

## Requirements

- Python 3.12+
- CUDA-capable GPU (optional, for faster inference)
- Docker (optional, for containerized deployment)

## License

[MIT License](LICENSE)