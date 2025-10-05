# Celara: Celestial Clarity

**Multi-Modal ResNet Exoplanet Classifier - NASA Space Challenge 2025–26**

A machine learning pipeline for exoplanet detection using Kepler mission data, featuring multi-modal 1D ResNet architecture with uncertainty quantification. Built for the 36-hour hackathon challenge with focus on scientific credibility and real-world applicability.

---

## Current Status: Steps 1-3 Complete (Ahead of Schedule)

**Data Pipeline:** Complete with 9,564 KOI entries and robust light curve processing  
**Feature Extraction:** Multi-modal features ready for neural network training  
**Visualization:** Comprehensive analysis and debugging tools implemented  
**Next:** Multi-modal ResNet implementation (Step 4)

---

## Implemented Features

### 1. **Data Loading & Management**
- **KOI Catalog Integration:** Direct NASA Exoplanet Archive access via TAP service
- **Lightkurve Integration:** Automated light curve download and caching
- **Robust Parameter Extraction:** Handles missing values with scientifically sound defaults

### 2. **Preprocessing Pipeline (`KeplerLCPreprocessor`)**
- **Median Normalization:** Flux standardization across targets
- **Savitzky-Golay Detrending:** 201-point window for stellar variability removal
- **NaN Handling:** Robust cleaning and gap management
- **Step-by-step Visualization:** Full processing pipeline transparency
- See [] for example of plots

### 3. **Feature Engineering (`KeplerFeatureExtractor`)**
- **Local View:** 201-bin phase-folded transit region (±2 transit durations)
- **Global View:** 2001-bin phase-folded full orbital cycle
- **Auxiliary Features:** Normalized stellar parameters (Teff, SNR, period, radius)
- **Total Features:** 2,206 per sample (201 + 2001 + 4)
- see [] for plots

---

## Architecture Overview

### **Planned Multi-Modal ResNet (Step 4)**
```
Input Features (2,206 total):
├── Local View (201): Phase-folded ±2 transit durations → 1D ResNet (3-5 blocks)
├── Global View (2001): Full orbital phase → 1D ResNet (5-8 blocks)  
└── Auxiliary (4): [Teff/6000, SNR/100, Period/365, Radius/10] → Dense layers

Late Fusion:
└── Concatenate feature vectors → Dense + MC Dropout → Sigmoid(p_planet)
```

### **Training Strategy**
- **Loss Function:** Focal Loss handles class imbalance
- **Optimizer:** AdamW with OneCycleLR scheduling
- **Uncertainty:** Monte Carlo Dropout (N=20-30 forward passes)
- **Primary Metric:** PR-AUC (Average Precision)

---

## Quick Start

### 1. Environment Setup
```bash
git clone https://github.com/kareem-xyz/nasa_25-26.git
cd nasa_25-26
pip install -r requirements.txt
```

### 2. Data Pipeline Demo
```python
# Load the main notebook
jupyter notebook Celara.ipynb

# Key classes for immediate use:
from Celara import KeplerLCPreprocessor, KeplerFeatureExtractor, extract_koi_parameters_from_row

# Process any Kepler light curve:
preprocessor = KeplerLCPreprocessor(detrend_window=201)
feature_extractor = KeplerFeatureExtractor(local_bins=201, global_bins=2001)

# Extract features for machine learning:
koi_params = extract_koi_parameters_from_row(koi_row)
features = feature_extractor.extract_features_from_koi_params(lc, koi_params)
# Returns: {'local_view': (201,), 'global_view': (2001,), 'aux_features': (4,)}
```

---


## Team

**Kareem** - NASA Space Challenge 2025–26 Solo Submission  
*Celara: Multi-Modal ResNet for Exoplanet Detection*

---

## References

- NASA Exoplanet Archive: [Kepler Q1-Q17 DR25 KOI Table](https://exoplanetarchive.ipac.caltech.edu/)
- Lightkurve: [Kepler/TESS Time Series Analysis](https://docs.lightkurve.org/)
- Original Astronet: [Shallue & Vanderburg (2018)](https://arxiv.org/abs/1712.05044)