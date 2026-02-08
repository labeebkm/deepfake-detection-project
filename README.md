# Deepfake Image Detection using Three-Stream EfficientNet

A robust deepfake image detection system built with TensorFlow/Keras, combining:
- RGB spatial features
- Frequency-domain features (DCT-based)
- Explicit forensic features

The project includes a modular training pipeline, inference API, evaluation scripts, and comprehensive Exploratory Data Analysis (EDA).

## Features

- **Three-stream EfficientNet** architecture (RGB + DCT frequency + explicit forensic features)
- **Attention-based stream fusion** for combining RGB, frequency, and explicit features
- **Comprehensive EDA** with automated artifact detection
- **Complete data pipeline** using tf.data API
- **Mixed precision training** (FP16) for GPU acceleration
- **REST API** using FastAPI
- **Grad-CAM visualization** for model explainability
- **TensorBoard integration** for monitoring

## Project Structure

```
deepfake-detection-tf/
|-- notebooks/              # Jupyter notebooks for EDA
|   |-- 01_data_exploration.ipynb
|   |-- 02_feature_analysis.ipynb
|   |-- 03_model_experiments.ipynb
|   `-- 04_results_analysis.ipynb
|-- eda/                    # EDA Python modules
|   |-- data_analyzer.py
|   |-- visualization.py
|   |-- statistical_tests.py
|   |-- artifact_detector.py
|   `-- report_generator.py
|-- configs/                # Configuration files
|   |-- config.yaml
|   `-- eda_config.json
|-- data/                   # Data pipeline
|   |-- dataset_loader.py
|   |-- preprocessing.py
|   |-- augmentation.py
|   `-- face_detection.py
|-- models/                 # Model architectures
|   |-- three_stream_net.py
|   |-- frequency_net.py
|   `-- model_factory.py
|-- training/               # Training utilities
|   |-- trainer.py
|   |-- losses.py
|   |-- callbacks.py
|   `-- metrics.py
|-- inference/              # Inference utilities
|   |-- detector.py
|   `-- api.py
|-- utils/                  # Utility functions
|   |-- visualization.py
|   |-- frequency_utils.py
|   `-- face_utils.py
|-- train.py                # Training script
|-- evaluate.py             # Evaluation script
|-- eda_report.py           # EDA report generation
|-- serve.py                # API server
|-- requirements.txt
|-- Dockerfile
`-- README.md
```


## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd deepfake-detection-tf
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install MTCNN for face detection:
```bash
pip install mtcnn
```

## Quick Start

### 1. Generate EDA Report

```bash
python eda_report.py --dataset faceforensics --data_dir ./data/raw --output ./reports --visualize --save
```

### 2. Train Model

```bash
python train.py --config configs/config.yaml --data_dir ./data/raw
```

### 3. Evaluate Model

```bash
python evaluate.py --model_path ./checkpoints/best_model.weights.h5 --data_dir ./data/raw/test
```

### 4. Serve API

```bash
python serve.py --model_path ./checkpoints/best_model.weights.h5 --config configs/config.yaml --host 0.0.0.0 --port 8000
```

## EDA Features

The EDA system provides:

1. **Dataset Statistics**: Class distribution, image counts, resolution analysis
2. **Visual Analysis**: Side-by-side comparisons, difference maps, frequency spectra
3. **Feature Analysis**: Frequency domain analysis, PCA/t-SNE visualization
4. **Artifact Detection**: Automated detection of face warping, lighting inconsistencies, blending artifacts
5. **Statistical Tests**: Kolmogorov-Smirnov, Chi-square, t-tests, ANOVA
6. **Automated Reports**: HTML/PDF reports with interactive visualizations

## Configuration

Edit `configs/config.yaml` to customize:
- Model architecture
- Training parameters
- Data augmentation
- Loss functions
- Callbacks

Edit `configs/eda_config.json` to customize EDA analysis options.

## Model Architecture

The three-stream architecture consists of:

1. **RGB Stream**: EfficientNet-B4 backbone pretrained on ImageNet
2. **Frequency Stream**: DCT-based frequency analysis network (FrequencyNet)
3. **Explicit Feature Stream**: MLP over hand-crafted forensic features (ELA, texture/gradient histograms, color moments)
4. **Fusion + Head**: attention over the three stream embeddings + binary classification

## Dataset Format

Expected directory structure:
```
data/raw/
  real/
    image1.jpg
    image2.jpg
  fake/
    image1.jpg
    image2.jpg
```

## API Usage

Once the API server is running, you can make predictions:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg"
```

Response:
```json
{
  "is_fake": false,
  "confidence": 0.95,
  "real_probability": 0.95,
  "fake_probability": 0.05
}
```

## Docker

Build and run with Docker:

```bash
docker build -t deepfake-detection .
docker run -p 8000:8000 deepfake-detection
```

## Git Setup

This project uses Git for version control. The repository includes:

- `.gitignore` - Excludes Python cache, virtual environments, model checkpoints, and other generated files
- `.gitattributes` - Ensures consistent line endings and handles binary files
- `CONTRIBUTING.md` - Guidelines for contributing to the project
- `LICENSE` - MIT License

### Initial Git Setup

If you're cloning this repository:

```bash
git clone <repository-url>
cd dfprojectv2
```

### Making Your First Commit

```bash
# Stage all files
git add .

# Commit with a descriptive message
git commit -m "Initial commit: Deepfake detection system with EDA"

# Push to remote (if you have a remote repository)
git push origin main
```

### Git Workflow

1. Create a branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit:
   ```bash
   git add .
   git commit -m "feat: Add new feature"
   ```

3. Push and create a Pull Request:
   ```bash
   git push origin feature/your-feature-name
   ```

See `CONTRIBUTING.md` for detailed contribution guidelines.

## License

MIT License - See LICENSE file for details

## Citation

If you use this code, please cite:

```bibtex
@software{deepfake_detection_three_stream,
  title  = {Deepfake Image Detection Using Three-Stream EfficientNet},
  author = {Labeeb K M},
  year   = {2025},
  url    = {https://github.com/labeebkm/deepfake-detection-project},
  note   = {RGB, frequency-domain, and explicit forensic feature fusion}
}
```
 To start the training
.\dfenv\python.exe serve.py --model_path .\checkpoints\best_model.weights.h5 --config configs\config.yaml

To test
curl.exe -F "file=@C:\Users\HP\Pictures\test.jpg" http://127.0.0.1:8000/predict

