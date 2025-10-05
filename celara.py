import lightkurve as lk
import numpy as np
# import statements
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import lightkurve as lk
from pathlib import Path
import os
import time
from typing import Optional
from tqdm.auto import tqdm

class KeplerLCPreprocessor:
    def __init__(self, detrend_window: int = 201):
        """
        Preprocess Kepler light curves from a local folder.
        
        Args:
            detrend_window (int): Window for Savitzky-Golay detrending
        """
        self.detrend_window = detrend_window

    def read_lc_file(self, filepath):
        """Read a single Kepler .fits file and return LightCurve."""
        return lk.KeplerLightCurve.read(filepath) # type: ignore
    
    def remove_nans(self, lc):
        """Remove NaN values from light curve."""
        return lc.remove_nans()
    
    def normalize_flux(self, lc):
        """Median normalize the flux."""
        median_flux = np.median(lc.flux)
        return lc / median_flux
    
    def detrend_flux(self, lc, window=None):
        """Apply Savitzky-Golay detrending."""
        if window is None:
            window = self.detrend_window
        window = min(window, len(lc.flux) // 2 * 2 + 1)  # Ensure odd
        trend = savgol_filter(lc.flux, window_length=window, polyorder=2)
        return lc / trend
    
    def extract_kic_id(self, file_path):
        """Extract KIC identifier from Kepler file path or filename."""
        if '/' in file_path:
            # Full path - use folder name
            return file_path.split('/')[-2].split('_')[0].replace('kplr', 'KIC ')
        else:
            # Just filename - extract directly
            return file_path.split('-')[0].replace('kplr', 'KIC ')

    def preprocess_single(self, lc, window=None):
        """
        Preprocess a single LightCurve object and return processed LightCurve.
        """
        # Step-by-step preprocessing
        lc_clean = self.remove_nans(lc)
        lc_normalized = self.normalize_flux(lc_clean)
        lc_detrended = self.detrend_flux(lc_normalized, window=window)
        return lc_detrended

    def preprocess_folder(self, folder_path: str, n_samples: Optional[int] = None):
        """
        Preprocess all or a subset of Kepler .fits files in a folder.

        Args:
            folder_path (str): Folder containing .fits files
            n_samples (Optional[int]): Number of files to process; None = all

        Returns:
            lightcurves: list of processed LightCurve objects
            filenames: list of str (names of processed files)
        """
        folder = Path(folder_path)
        fits_files = list(folder.rglob('*.fits'))

        if n_samples is not None:
            fits_files = fits_files[:n_samples]  # simple sampling

        lightcurves = []
        filenames = []

        for f in fits_files:
            try:
                lc = self.read_lc_file(f)
                lc_processed = self.preprocess_single(lc)
                lightcurves.append(lc_processed)
                filenames.append(f.name)
            except Exception as e:
                print(f"[WARN] Failed for {f.name}: {e}")

        return lightcurves, filenames



class KeplerFeatureExtractor: # Global, Local, AUX
    def __init__(self, local_bins: int = 201, global_bins: int = 2001):
        """
        Extract features from Kepler light curves for machine learning.
        
        Args:
            local_bins (int): Number of bins for local view (±2 transit durations)
            global_bins (int): Number of bins for global view (full orbit)
        """
        self.local_bins = local_bins
        self.global_bins = global_bins

    def phase_fold(self, time, flux, period, epoch=0):
        """Phase-fold light curve."""
        phase = ((time - epoch) / period) % 1.0
        phase[phase > 0.5] -= 1.0
        return phase, flux

    def bin_phase_curve(self, phase, flux, n_bins):
        """Bin phase-folded curve."""
        bins = np.linspace(-0.5, 0.5, n_bins + 1)
        binned_flux = []
        for i in range(n_bins):
            mask = (phase >= bins[i]) & (phase < bins[i+1])
            if np.any(mask):
                binned_flux.append(np.median(flux[mask]))
            else:
                binned_flux.append(1.0)
        return np.array(binned_flux)

    def extract_auxiliary_features(self, period, aux_features=None):
        """
        Extract and normalize auxiliary features.
        
        Args:
            period: Orbital period in days
            aux_features: Dict with stellar parameters (Teff, SNR, radius, etc.)
            
        Returns:
            Normalized auxiliary feature array
        """
        # Default values if no aux_features provided
        aux = np.array([1.0, 1.0, 1.0, 1.0])
        
        if aux_features:
            aux = np.array([
                aux_features.get('teff', 5000) / 6000,      # Normalized temperature
                aux_features.get('snr', 10) / 100,          # Normalized SNR
                period / 365,                               # Period in years
                aux_features.get('radius', 1) / 10          # Normalized radius
            ])
        
        return aux

    def extract_local_view(self, phase, flux, transit_duration, period):
        """
        Extract local view around transit (±2 transit durations).
        
        Args:
            phase: Phase-folded time array
            flux: Phase-folded flux array
            transit_duration: Transit duration in days
            period: Orbital period in days
            
        Returns:
            Binned local view array
        """
        # Local view: ±2 transit durations around transit
        transit_width = 2 * transit_duration / period  # Phase units
        local_mask = np.abs(phase) <= transit_width
        
        if np.any(local_mask):
            local_phase = phase[local_mask] / transit_width * 0.5  # Normalize to ±0.5
            local_flux = flux[local_mask]
            local_view = self.bin_phase_curve(local_phase, local_flux, self.local_bins)
        else:
            local_view = np.ones(self.local_bins)
            
        return local_view

    def extract_global_view(self, phase, flux):
        """
        Extract global view of full orbital phase.
        
        Args:
            phase: Phase-folded time array
            flux: Phase-folded flux array
            
        Returns:
            Binned global view array
        """
        return self.bin_phase_curve(phase, flux, self.global_bins)

    def extract_features(self, time, flux, period, transit_duration, aux_features=None):
        """
        Extract complete feature set: local view, global view, and auxiliary features.
        
        Args:
            time: Time array
            flux: Detrended flux array  
            period: Orbital period in days
            transit_duration: Transit duration in days
            aux_features: Dict with stellar parameters (Teff, SNR, radius, etc.)
        
        Returns:
            dict with 'local_view', 'global_view', 'aux_features'
        """
        # Phase-fold the light curve
        phase, flux_folded = self.phase_fold(time, flux, period)
        
        # Extract different feature views
        global_view = self.extract_global_view(phase, flux_folded)
        local_view = self.extract_local_view(phase, flux_folded, transit_duration, period)
        aux_features_array = self.extract_auxiliary_features(period, aux_features)
        
        return {
            'local_view': local_view,
            'global_view': global_view, 
            'aux_features': aux_features_array
        }

    def extract_features_from_lc(self, lc, period, transit_duration, aux_features=None):
        """
        Convenience method to extract features directly from LightCurve object.
        
        Args:
            lc: LightCurve object (preprocessed)
            period: Orbital period in days
            transit_duration: Transit duration in days
            aux_features: Dict with stellar parameters
            
        Returns:
            dict with 'local_view', 'global_view', 'aux_features'
        """
        return self.extract_features(
            time=lc.time.value,
            flux=lc.flux.value,
            period=period,
            transit_duration=transit_duration,
            aux_features=aux_features
        )

    def extract_features_from_koi_params(self, lc, koi_params):
        """
        NEW: Extract features directly from KOI parameter dictionary.
        
        Args:
            lc: LightCurve object (preprocessed)
            koi_params: Parameter dictionary from extract_koi_parameters_from_row()
            
        Returns:
            dict with 'local_view', 'global_view', 'aux_features'
        """
        return self.extract_features_from_lc(
            lc=lc,
            period=koi_params['period'],
            transit_duration=koi_params['duration'],
            aux_features=koi_params['aux_features']
        )

    def extract_features_batch(self, lightcurves, periods, transit_durations, aux_features_list=None):
        """
        Extract features from multiple light curves.
        
        Args:
            lightcurves: List of LightCurve objects
            periods: List of orbital periods
            transit_durations: List of transit durations
            aux_features_list: List of auxiliary feature dicts (optional)
            
        Returns:
            List of feature dictionaries
        """
        if aux_features_list is None:
            aux_features_list = [None] * len(lightcurves)
            
        features_list = []
        for lc, period, duration, aux in zip(lightcurves, periods, transit_durations, aux_features_list):
            features = self.extract_features_from_lc(lc, period, duration, aux)
            features_list.append(features)
            
        return features_list

    def extract_features_batch_from_koi(self, lightcurves, koi_params_list):
        """
        NEW: Extract features from multiple light curves using KOI parameter dictionaries.
        
        Args:
            lightcurves: List of LightCurve objects
            koi_params_list: List of parameter dictionaries from extract_koi_parameters_from_row()
            
        Returns:
            List of feature dictionaries
        """
        features_list = []
        for lc, koi_params in zip(lightcurves, koi_params_list):
            features = self.extract_features_from_koi_params(lc, koi_params)
            features_list.append(features)
            
        return features_list

    def get_feature_shapes(self):
        """Return the expected shapes of extracted features."""
        return {
            'local_view': (self.local_bins,),
            'global_view': (self.global_bins,),
            'aux_features': (4,)
        }


# -----------------------------
# Example usage with NEW methods
# -----------------------------
# feature_extractor = KeplerFeatureExtractor(local_bins=201, global_bins=2001)
# 
# # Method 1: Traditional approach
# features = feature_extractor.extract_features_from_lc(lc, period=3.52, transit_duration=0.15)
# 
# # Method 2: NEW - Using KOI parameter dictionary
# koi_params = extract_koi_parameters_from_row(koi_row)
# features = feature_extractor.extract_features_from_koi_params(lc, koi_params)
# 
# # Method 3: NEW - Batch processing with KOI parameters
# features_list = feature_extractor.extract_features_batch_from_koi(lightcurves, koi_params_list)