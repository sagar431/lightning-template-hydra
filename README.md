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

## Web Interface

Our modern web interface provides an intuitive way to interact with the model:

![Web Interface Demo](./assets/web_interface.png)

### Key Features
- **Drag & Drop**: Easy image upload with drag-and-drop functionality
- **Real-time Processing**: Instant predictions using GPU acceleration
- **Confidence Scores**: Visual progress bars showing prediction confidence
- **Top 5 Predictions**: See alternative breed predictions with confidence levels
- **Mobile Responsive**: Works seamlessly on all devices

Example prediction:
```
Top 5 Predictions:
1. Boxer            (99.99% confidence) ████████████████████ 
2. Bulldog          (0.01% confidence)  █
3. German Shepherd  (0.00% confidence)  
4. Rottweiler       (0.00% confidence)  
5. Golden Retriever (0.00% confidence)  
```

## Dataset

This project uses a dog breed classification dataset hosted on Google Drive. The data will be automatically downloaded and organized in the following structure:

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

### Dataset Setup

1. **Automatic Download**:
   ```bash
   # The dataset will be automatically downloaded on first run
   python src/train.py
   ```

2. **Manual Download**:
   - Download the dataset from our Google Drive link
   - Extract to `data/dog_breed/dataset/`
   - Ensure the directory structure matches the above layout

### Data Preprocessing
- Images are automatically resized to model input size
- Random horizontal flips and rotations for augmentation
- Normalization using ImageNet statistics
- Train/validation split handled by directory structure

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

## Model Architecture

Based on our proven configuration:
- **Base Model**: ResNet18 from `timm` with pretrained weights
- **Training**: 
  - Loss: CrossEntropyLoss
  - Optimizer: Adam (lr=1e-3)
  - Batch Size: 32
  - Epochs: 10
- **Data Augmentation**:
  - Random horizontal flips
  - Random rotations
  - Normalization
- **Validation**: Best checkpoints saved based on accuracy

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