# Celara: Kepler Exoplanet Detection Pipeline

Professional machine learning pipeline for exoplanet detection using Kepler mission data with multi-modal ResNet CNN architecture and Flask web interface.

## Current Status

✅ **Data Pipeline**: KOI catalog loading, balanced sampling, lightcurve download  
✅ **Preprocessing**: Normalization, detrending, robust feature extraction  
✅ **Model Architecture**: 2M parameter ResNet CNN with multi-modal inputs  
✅ **Web Interface**: Flask application with file upload and batch processing  
✅ **Production Ready**: Clean codebase, error handling, deployment ready  

## Features

- **Automated Data Loading**: NASA Exoplanet Archive integration via TAP service
- **Robust Preprocessing**: Median normalization, Savitzky-Golay detrending, NaN handling  
- **Multi-Modal Architecture**: 
  - Local view: 201 bins (±2 transit durations) - fine-grained transit analysis
  - Global view: 2001 bins (full orbital phase) - long-term variability patterns
  - Auxiliary: 4 stellar parameters - host star characteristics
- **Enhanced ResNet CNN**: 2M parameters with deep feature extraction
- **Web Interface**: Professional Flask application with file upload support
- **Production Features**: Model persistence, batch processing, visualization

## Quick Start

### Training Pipeline
```bash
git clone https://github.com/kareem-xyz/nasa_25-26.git
cd nasa_25-26
pip install -r requirements.txt
jupyter notebook Celara.ipynb
```

### Web Interface
```bash
cd flask
python app.py
# Navigate to http://localhost:5000
```

## Model Architecture

**Enhanced ResNet CNN (~2M Parameters)**
- **Local Branch**: Deep ResNet (64→128→256 filters) on transit features
- **Global Branch**: Deep ResNet (64→128→256 filters) on orbital features  
- **Auxiliary Branch**: Dense network (64→128→256) on stellar parameters
- **Feature Fusion**: 768-dimensional combined representation
- **Classification**: Deep dense layers (512→256→128→1) with dropout

## API Usage

```python
from celara_model import create_simple_resnet_trimodal
from celara_utils import process_kepler_dataset

# Process lightcurve dataset
X, y = process_kepler_dataset(df, lightcurve_dir, max_samples=1000)
# Returns: X.shape = (n_samples, 2206), y.shape = (n_samples,)

# Create enhanced model
model = create_simple_resnet_trimodal()
model.summary()  # ~2M parameters

# Training configuration
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall', 'auc']
)
```

## Project Structure

```
├── Celara.ipynb              # Main training pipeline
├── celara.py                 # Core preprocessing classes
├── celara_model.py           # Enhanced ResNet architecture
├── celara_utils.py           # Processing utilities
├── model_interface.py        # Model loading and inference
├── flask/
│   ├── app.py               # Web application
│   ├── templates/           # HTML templates
│   └── static/              # CSS, JS, images
├── models/                  # Trained model storage
└── data/                    # KOI catalog and lightcurves
```

## Web Interface Features

- **File Upload**: Support for CSV files, FITS lightcurves, KOI ID lookup
- **Batch Processing**: Handle multiple targets efficiently
- **Visualization**: Interactive plots and classification results
- **Model Info**: Architecture details and performance metrics
- **Download Results**: Export predictions and visualizations

## Performance

- **Architecture**: Multi-modal ResNet CNN with 2M parameters
- **Training**: 50 epochs with early stopping and learning rate scheduling
- **Evaluation**: Comprehensive metrics including ROC-AUC and precision-recall
- **Deployment**: Flask web interface with model persistence