"""
Celara Model Interface
======================
Interface between Flask app and Celara model
Provides prediction, visualization, and model info functions
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from pathlib import Path
import tensorflow as tf
import keras
from celara_model import create_simple_resnet_trimodal

# Global model variable
_model = None
_model_loaded = False

def load_trained_model():
    """
    Load the trained Celara model
    
    Returns:
        bool: True if model loaded successfully, False otherwise
    """
    global _model, _model_loaded
    
    # Try to load from models/keras directory
    model_paths = [
        'models/keras/best_exoplanet_model.keras',
        'models/keras/celara_model.keras',
        'celara_model.keras',
        'best_exoplanet_model.keras'
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                _model = keras.models.load_model(model_path)
                _model_loaded = True
                print(f"Model loaded from: {model_path}")
                return True
            except Exception as e:
                print(f"Failed to load model from {model_path}: {e}")
                continue
    
    # If no saved model found, create architecture for demo
    try:
        _model = create_simple_resnet_trimodal()
        _model_loaded = False  # Mark as demo mode
        print("Using demo model (architecture only)")
        return False
    except Exception as e:
        print(f"Failed to create demo model: {e}")
        return False


def get_model_info():
    """
    Get information about the loaded model
    
    Returns:
        dict: Model information including loaded status, architecture, parameters
    """
    global _model, _model_loaded
    
    if _model is None:
        load_trained_model()
    
    info = {
        'loaded': _model_loaded,
        'architecture': 'AstroNet_ResNet_Lightweight' if _model else None,
        'parameters': None
    }
    
    if _model:
        try:
            # info['parameters'] = _model.count_params()
            info['parameters'] = 'Available'
        except:
            info['parameters'] = None
    
    return info


def predict_batch(df):
    """
    Run batch predictions on DataFrame
    
    Args:
        df: DataFrame with orbital parameters
        
    Returns:
        DataFrame with predictions and confidence scores
    """
    global _model, _model_loaded
    
    if _model is None:
        load_trained_model()
    
    # For demo mode or if model not loaded, return random predictions
    if not _model_loaded:
        n_samples = len(df)
        predictions = np.random.random(n_samples)
        # Confidence based on distance from decision boundary (0.5)
        confidence = 1.0 - 2 * np.abs(predictions - 0.5)
        
        results_df = df.copy()
        results_df['prediction'] = predictions
        results_df['confidence'] = confidence
        
        return results_df
    
    # TODO: Implement actual model prediction
    # This would require feature extraction from the orbital parameters
    # For now, return demo predictions
    n_samples = len(df)
    predictions = np.random.random(n_samples)
    # Confidence based on distance from decision boundary (0.5)
    confidence = 1.0 - 2 * np.abs(predictions - 0.5)
    
    results_df = df.copy()
    results_df['prediction'] = predictions
    results_df['confidence'] = confidence
    
    return results_df


def create_visualizations(results_df, save_dir, timestamp):
    """
    Create visualization plots for results
    
    Args:
        results_df: DataFrame with prediction results
        save_dir: Directory to save plots
        timestamp: Unique timestamp for file naming
        
    Returns:
        dict: Paths to generated plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create a combined plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Celara Exoplanet Classification Results', fontsize=16)
    
    predictions = results_df['prediction'].values
    confidence = results_df['confidence'].values
    
    # Plot 1: Prediction distribution
    axes[0, 0].hist(predictions, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Planet Probability')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Prediction Distribution')
    axes[0, 0].axvline(x=0.5, color='red', linestyle='--', label='Decision Threshold')
    axes[0, 0].legend()
    
    # Plot 2: Confidence distribution
    axes[0, 1].hist(confidence, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].set_xlabel('Prediction Confidence')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Confidence Distribution')
    
    # Plot 3: Prediction vs Confidence scatter
    scatter = axes[1, 0].scatter(predictions, confidence, alpha=0.6, c=predictions, cmap='viridis')
    axes[1, 0].set_xlabel('Planet Probability')
    axes[1, 0].set_ylabel('Confidence')
    axes[1, 0].set_title('Prediction vs Confidence')
    plt.colorbar(scatter, ax=axes[1, 0])
    
    # Plot 4: Top candidates
    top_candidates = results_df.nlargest(10, 'prediction')
    y_pos = np.arange(len(top_candidates))
    axes[1, 1].barh(y_pos, top_candidates['prediction'], alpha=0.7, color='gold')
    axes[1, 1].set_yticks(y_pos)
    axes[1, 1].set_yticklabels([str(tid)[:10] for tid in top_candidates['target_id']])
    axes[1, 1].set_xlabel('Planet Probability')
    axes[1, 1].set_title('Top 10 Candidates')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_dir, f'results_{timestamp}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'combined': plot_path
    }


# Initialize model on import
load_trained_model()