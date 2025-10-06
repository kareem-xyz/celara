# Function to fetch and cache light curves using Lightkurve
import os
from pathlib import Path
import lightkurve as lk
import pandas as pd
import time
import numpy as np
from tqdm.auto import tqdm

from celara import KeplerFeatureExtractor, KeplerLCPreprocessor


def fetch_lightcurves(target_list, download_dir="data/lightcurves", mission="Kepler",
                      overwrite=False, sleep=0.5):
    """
    Downloads and caches light curves for a list of target IDs (KIC/EPIC/TIC) sequentially.
    Skips already-downloaded files unless overwrite=True. Returns a DataFrame summarizing success/failure.
    """
    os.makedirs(download_dir, exist_ok=True)
    results = []

    for target in tqdm(target_list, desc=f"Fetching {mission} light curves"):
        tstr = str(target).strip()
        path = Path(download_dir) / f"{tstr}.fits"
        if path.exists() and not overwrite:
            results.append({"target_id": tstr, "filepath": str(path), "success": True, "error": None})
            continue

        search_id = f"KIC {tstr}" if mission=="Kepler" else f"{mission} {tstr}"
        try:
            search = lk.search_lightcurve(search_id)
            if len(search) == 0:
                results.append({"target_id": tstr, "filepath": None, "success": False, "error": "Not found"})
                continue

            lc = search[0].download(download_dir=str(download_dir))
            if lc is None:
                results.append({"target_id": tstr, "filepath": None, "success": False, "error": "Download returned None"})
                continue

            results.append({"target_id": tstr, "filepath": str(path), "success": True, "error": None})

        except Exception as e:
            results.append({"target_id": tstr, "filepath": None, "success": False, "error": str(e)})

        time.sleep(sleep)

    return pd.DataFrame(results)


def extract_koi_parameters_from_row(koi_row):
    """
    Extract KOI parameters from a KOI DataFrame row with fallback defaults.
    
    Args:
        koi_row: Single row from KOI DataFrame
        
    Returns:
        dict with period, duration, disposition, aux_features, data_quality
    """
    # Extract with defaults
    period = koi_row.get('koi_period', 3.0) if pd.notna(koi_row.get('koi_period')) else 3.0
    duration = koi_row.get('koi_duration', 0.15) if pd.notna(koi_row.get('koi_duration')) else 0.15
    disposition = koi_row.get('koi_disposition', 'UNKNOWN')
    koi_name = koi_row.get('kepoi_name', f"KOI-{koi_row.get('kepid', 'unknown')}")
    
    # Auxiliary features with defaults
    aux_features = {
        'teff': koi_row.get('koi_seff', 5000) if pd.notna(koi_row.get('koi_seff')) else 5000,
        'snr': koi_row.get('koi_depth', 10) if pd.notna(koi_row.get('koi_depth')) else 10,
        'radius': koi_row.get('koi_srad', 1.0) if pd.notna(koi_row.get('koi_srad')) else 1.0
    }
    
    # Track what was original vs default
    data_quality = {
        'period_original': pd.notna(koi_row.get('koi_period')),
        'duration_original': pd.notna(koi_row.get('koi_duration')),
        'teff_original': pd.notna(koi_row.get('koi_seff')),
        'snr_original': pd.notna(koi_row.get('koi_depth')),
        'radius_original': pd.notna(koi_row.get('koi_srad'))
    }
    
    return {
        'period': period,
        'duration': duration,
        'disposition': disposition,
        'koi_name': koi_name,
        'aux_features': aux_features,
        'data_quality': data_quality
    }

# UNIFIED PROCESSING FUNCTION
def process_kepler_dataset(df, lightcurve_dir, max_samples=None, path="data", save_name="features"):
    """
    Efficiently process Kepler lightcurve dataset with full optimizations.
    
    Args:
        df: DataFrame with KOI data (from balanced_subset or any KOI dataframe)
        lightcurve_dir: Path to directory containing .fits files
        max_samples: Maximum samples to process (None = all)
        save_name: Name for saved .npz file
        
    Returns:
        X: Feature array (n_samples, 2206)
        y: Label array (n_samples,)
    """
    
    # Initialize processors
    preprocessor = KeplerLCPreprocessor(detrend_window=201)
    feature_extractor = KeplerFeatureExtractor(local_bins=201, global_bins=2001)
    
    # Limit samples if specified
    if max_samples:
        df = df.head(max_samples)
    
    print(f"Processing {len(df)} samples from dataset...")
    
    # Build file index once (major optimization)
    file_index = {}
    if os.path.exists(lightcurve_dir):
        all_fits = list(Path(lightcurve_dir).rglob('*.fits'))
        for f in tqdm(all_fits, desc="Building file index"):
            try:
                kic_num = int(f.name.split('-')[0].replace('kplr', '').lstrip('0'))
                file_index[kic_num] = f
            except:
                continue
        print(f"File index: {len(file_index)} files")
    else:
        print(f"Lightcurve directory not found: {lightcurve_dir}")
        return None, None
    
    # Filter to available files only
    available_mask = df['kepid'].isin(file_index.keys())
    df_available = df[available_mask].copy()
    print(f"Available lightcurves: {len(df_available)}/{len(df)}")
    
    if len(df_available) == 0:
        print("No matching lightcurve files found")
        return None, None
    
    # Process all available samples
    X_list = []
    y_list = []
    
    for _, row in tqdm(df_available.iterrows(), total=len(df_available), desc="Processing lightcurves"):
        try:
            kic_number = row['kepid']
            file_path = file_index[kic_number]
            
            # Extract parameters and process
            koi_params = extract_koi_parameters_from_row(row)
            lc_raw = preprocessor.read_lc_file(file_path)
            lc_processed = preprocessor.preprocess_single(lc_raw)
            features = feature_extractor.extract_features_from_koi_params(lc_processed, koi_params)
            
            # Create feature vector (2206 total)
            feature_vector = np.concatenate([
                features['local_view'],    # 201
                features['global_view'],   # 2001  
                features['aux_features']   # 4
            ])
            label = 1 if koi_params['disposition'] == 'CONFIRMED' else 0
            
            X_list.append(feature_vector)
            y_list.append(label)
            
        except Exception as e:
            continue  # Skip failed samples
    
    if X_list:
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Save results
        save_path = f'{path}/{save_name}.npz'
        np.savez(save_path, X=X, y=y)
        
        print(f"Successfully processed {len(X)} samples")
        print(f"   Shape: {X.shape}")
        print(f"   Labels: {np.bincount(y)} (0=FP, 1=Confirmed)")
        print(f"   Saved: {save_path}")
        
        return X, y
    else:
        print("No samples processed successfully")
        return None, None
