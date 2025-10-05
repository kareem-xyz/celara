# Celara: Kepler Exoplanet Detection Pipeline

Machine learning pipeline for exoplanet detection using Kepler mission data with multi-modal feature extraction and ResNet classification.

## Current Status

✅ **Data Pipeline**: KOI catalog loading, balanced sampling, lightcurve download  
✅ **Preprocessing**: Normalization, detrending, NaN handling  
✅ **Feature Extraction**: 2206-feature pipeline (local + global + auxiliary)  

## Features

- **Automated Data Loading**: NASA Exoplanet Archive integration via TAP
- **Robust Preprocessing**: Median normalization, Savitzky-Golay detrending  
- **Multi-Modal Features**: 
  - Local view: 201 bins (±2 transit durations)
  - Global view: 2001 bins (full orbital phase)
  - Auxiliary: 4 stellar parameters
- **Lightweight Architecture**: 71K parameters with Global Average Pooling
- **Optimized Processing**: File indexing, batch processing, progress tracking

## Quick Start

```bash
git clone https://github.com/kareem-xyz/nasa_25-26.git
cd nasa_25-26
pip install -r requirements.txt
jupyter notebook Celara.ipynb
```

## Usage

```python
from celara import KeplerLCPreprocessor, KeplerFeatureExtractor
from celara_model import create_astronet_resnet_trimodal
from celara_utils import process_kepler_dataset

# Process lightcurve dataset
X, y = process_kepler_dataset(df, lightcurve_dir, max_samples=1000)
# Returns: X.shape = (n_samples, 2206), y.shape = (n_samples,)

# Create and train model
model = create_astronet_resnet_trimodal()
# 71K parameters, 3 inputs: local + global + auxiliary features
```

## Architecture

- **Local Branch**: 1D ResNet on 201-bin transit features
- **Global Branch**: 1D ResNet on 2001-bin orbital features  
- **Auxiliary Branch**: Dense layers on stellar parameters
- **Fusion**: Concatenate + Dense layers + Dropout
- **Optimization**: Global Average Pooling for parameter efficiency

## Files

- `Celara.ipynb`: Main pipeline notebook with complete training pipeline
- `celara.py`: Core preprocessing and feature extraction classes
- `celara_model.py`: Lightweight AstroNet+ResNet model architecture
- `celara_utils.py`: Utility functions and processing pipeline
- `data/`: KOI catalog and lightcurve files