"""Celara Flask Application - Web interface for exoplanet classification"""

from flask import Flask, render_template, request, jsonify, send_file
import os
import sys
from werkzeug.utils import secure_filename
from datetime import datetime
import pandas as pd
import numpy as np
import tempfile
import zipfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import model functions from separate file
from model_interface import (
    load_trained_model,
    predict_batch,
    create_visualizations,
    get_model_info
)

# Import Celara processing functions
from celara_utils import fetch_lightcurves, process_kepler_dataset
from celara import KeplerLCPreprocessor, KeplerFeatureExtractor

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'U6RC75FUVFU6R75CR56E4DF65R5CRTUCTUTC86VR57RC6DCEDCE5D5E'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['LIGHTCURVE_FOLDER'] = 'lightcurves'
app.config['RESULTS_FOLDER'] = 'static/results'
app.config['PLOTS_FOLDER'] = 'static/plots'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'txt', 'fits', 'fit'}

# Ensure directories exist
for folder in [app.config['UPLOAD_FOLDER'], 
               app.config['LIGHTCURVE_FOLDER'],
               app.config['RESULTS_FOLDER'], 
               app.config['PLOTS_FOLDER']]:
    os.makedirs(folder, exist_ok=True)


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def process_koi_ids(koi_text, timestamp):
    """
    Process single KOI ID: download light curve and extract features
    
    Args:
        koi_text: String containing single KOI ID
        timestamp: Unique identifier for this processing session
        
    Returns:
        DataFrame with predictions
    """
    # Parse single KOI ID
    koi_text = koi_text.strip()
    koi_id = None
    
    # Extract numeric part from KOI-123 or KIC-456 format
    if koi_text.upper().startswith('KOI'):
        koi_id = koi_text.split('-')[-1] if '-' in koi_text else koi_text[3:]
    elif koi_text.upper().startswith('KIC'):
        koi_id = koi_text.split('-')[-1] if '-' in koi_text else koi_text[3:]
    elif koi_text.isdigit():
        koi_id = koi_text
    
    if not koi_id:
        raise ValueError("Invalid KOI/KIC ID format")
    
    # Download light curve
    download_dir = os.path.join(app.config['LIGHTCURVE_FOLDER'], f"session_{timestamp}")
    os.makedirs(download_dir, exist_ok=True)
    
    download_results = fetch_lightcurves([koi_id], download_dir=download_dir)
    successful_downloads = download_results[download_results['success'] == True]
    
    if len(successful_downloads) == 0:
        raise ValueError(f"Failed to download light curve for {koi_text}")
    
    # Create mock KOI DataFrame for processing
    koi_df = pd.DataFrame({
        'kepid': [int(koi_id)],
        'koi_period': [3.0],  # Default values
        'koi_duration': [0.15],
        'koi_disposition': ['CANDIDATE']
    })
    
    # Process with Celara
    X, y = process_kepler_dataset(
        df=koi_df,
        lightcurve_dir=download_dir,
        max_samples=1,
        path=download_dir,
        save_name=f"koi_features_{timestamp}"
    )
    
    if X is None:
        raise ValueError("Failed to process light curve")
    
    # Create results DataFrame
    prediction = np.random.random()  # TODO: Replace with actual model prediction
    confidence = 1.0 - 2 * np.abs(prediction - 0.5)  # Confidence based on distance from 0.5
    results_df = pd.DataFrame({
        'target_id': [f"KIC-{koi_id}"],
        'prediction': [prediction],
        'confidence': [confidence]
    })
    
    return results_df


def process_lightcurve_file(file, timestamp):
    """
    Process single uploaded FITS light curve file
    
    Args:
        file: Single uploaded FITS file
        timestamp: Unique identifier for this processing session
        
    Returns:
        DataFrame with predictions
    """
    # Save uploaded file
    upload_dir = os.path.join(app.config['LIGHTCURVE_FOLDER'], f"upload_{timestamp}")
    os.makedirs(upload_dir, exist_ok=True)
    
    if not (file.filename.lower().endswith('.fits') or file.filename.lower().endswith('.fit')):
        raise ValueError("File must be a FITS file")
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(upload_dir, filename)
    file.save(filepath)
    
    # Initialize Celara processors
    preprocessor = KeplerLCPreprocessor()
    feature_extractor = KeplerFeatureExtractor()
    
    # Process the light curve
    try:
        # Extract target ID from filename
        target_id = Path(filepath).stem
        
        # Read and preprocess light curve
        lc_raw = preprocessor.read_lc_file(filepath)
        lc_processed = preprocessor.preprocess_single(lc_raw)
        
        # Extract features with default parameters
        features = feature_extractor.extract_features_from_lc(
            lc_processed, 
            period=3.0,  # Default period
            transit_duration=0.15  # Default duration
        )
        
        # TODO: Replace with actual model prediction
        prediction = np.random.random()
        confidence = 1.0 - 2 * np.abs(prediction - 0.5)  # Confidence based on distance from 0.5
        
        results_df = pd.DataFrame({
            'target_id': [target_id],
            'prediction': [prediction],
            'confidence': [confidence]
        })
        
        return results_df
        
    except Exception as e:
        raise ValueError(f"Failed to process light curve file: {str(e)}")


def process_lightcurve_files(files, timestamp):
    """
    Process uploaded FITS light curve files
    
    Args:
        files: List of uploaded FITS files
        timestamp: Unique identifier for this processing session
        
    Returns:
        DataFrame with predictions
    """
    # Save uploaded files
    upload_dir = os.path.join(app.config['LIGHTCURVE_FOLDER'], f"upload_{timestamp}")
    os.makedirs(upload_dir, exist_ok=True)
    
    saved_files = []
    for file in files:
        if allowed_file(file.filename) and file.filename.lower().endswith(('.fits', '.fit')):
            filename = secure_filename(file.filename)
            filepath = os.path.join(upload_dir, filename)
            file.save(filepath)
            saved_files.append(filepath)
    
    if not saved_files:
        raise ValueError("No valid FITS files found")
    
    # Initialize Celara processors
    preprocessor = KeplerLCPreprocessor()
    feature_extractor = KeplerFeatureExtractor()
    
    # Process each light curve
    results = []
    for filepath in saved_files:
        try:
            # Extract target ID from filename
            target_id = Path(filepath).stem
            
            # Read and preprocess light curve
            lc_raw = preprocessor.read_lc_file(filepath)
            lc_processed = preprocessor.preprocess_single(lc_raw)
            
            # Extract features with default parameters
            features = feature_extractor.extract_features_from_lc(
                lc_processed, 
                period=3.0,  # Default period
                transit_duration=0.15  # Default duration
            )
            
            # TODO: Replace with actual model prediction
            prediction = np.random.random()
            confidence = 1.0 - 2 * np.abs(prediction - 0.5)  # Confidence based on distance from 0.5
            
            results.append({
                'target_id': target_id,
                'prediction': prediction,
                'confidence': confidence
            })
            
        except Exception as e:
            app.logger.warning(f"Failed to process {filepath}: {str(e)}")
            continue
    
    if not results:
        raise ValueError("Failed to process any light curve files")
    
    return pd.DataFrame(results)


def validate_csv_columns(df):
    """Validate that CSV has required columns"""
    required_cols = [
        'target_id', 'period', 'epoch', 'duration',
        'stellar_radius', 'teff', 'snr'
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    return True


@app.route('/')
def index():
    """Home page with upload interface"""
    model_info = get_model_info()
    return render_template('index.html', model_loaded=model_info['loaded'])


@app.route('/about')
def about():
    """About page with project and creator information"""
    model_info = get_model_info()
    return render_template('about.html', model_loaded=model_info['loaded'])


@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle file upload and process predictions for multiple input types
    
    Supports:
    - CSV files with orbital parameters
    - FITS light curve files
    - KOI ID lookup and download
    
    Returns:
        JSON with prediction results, statistics, and visualization paths
    """
    
    try:
        mode = request.form.get('mode', 'csv')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = None  # Initialize filename for all modes
        
        # Process based on input mode
        if mode == 'csv':
            # Validate file upload
            if 'file' not in request.files:
                return jsonify({'error': 'No file uploaded'}), 400
            
            file = request.files['file']
            
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if not allowed_file(file.filename) or not (file.filename and file.filename.lower().endswith(('.csv', '.txt'))):
                return jsonify({
                    'error': 'Invalid file type. Please upload CSV or TXT file'
                }), 400
            
            # Read and validate CSV
            df = pd.read_csv(file.stream)
            validate_csv_columns(df)
            
            # Save original file
            filename = secure_filename(file.filename or "uploaded_file")
            upload_path = os.path.join(
                app.config['UPLOAD_FOLDER'], 
                f"{timestamp}_{filename}"
            )
            df.to_csv(upload_path, index=False)
            
            # Run predictions
            results_df = predict_batch(df)
            
        elif mode == 'lightcurve':
            # Get uploaded FITS file
            if 'file' not in request.files:
                return jsonify({'error': 'No file uploaded'}), 400
            
            file = request.files['file']
            
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            # Process single light curve file
            results_df = process_lightcurve_file(file, timestamp)
            filename = secure_filename(file.filename or "lightcurve_file.fits")
            
        elif mode == 'koi':
            # Get single KOI ID from form
            koi_text = request.form.get('koi_id', '').strip()
            
            if not koi_text:
                return jsonify({'error': 'No KOI ID provided'}), 400
            
            # Process single KOI ID
            results_df = process_koi_ids(koi_text, timestamp)
            filename = f"koi_{koi_text.replace('-', '_').replace(' ', '_')}.csv"
            
        else:
            return jsonify({'error': f'Unknown mode: {mode}'}), 400
        
        # Generate visualizations
        plot_paths = create_visualizations(
            results_df,
            save_dir=app.config['PLOTS_FOLDER'],
            timestamp=timestamp
        )
        
        # Calculate summary statistics
        predictions = np.array(results_df['prediction'].values)
        confidence_scores = np.array(results_df['confidence'].values)
        
        summary = {
            'total_targets': int(len(results_df)),
            'planet_candidates': int(np.sum(predictions > 0.5)),
            'high_confidence': int(
                np.sum((predictions > 0.7) & (confidence_scores > 0.6))
            ),
            'avg_confidence': float(np.mean(confidence_scores))
        }
        
        # Get all candidates sorted by prediction probability
        all_candidates = results_df.sort_values('prediction', ascending=False)[[
            'target_id', 'prediction', 'confidence'
        ]].to_dict('records')
        
        # Get top 5 candidates for detailed view
        top_candidates = results_df.nlargest(5, 'prediction')[[
            'target_id', 'prediction', 'confidence'
        ]].to_dict('records')
        
        # Save results CSV
        results_path = os.path.join(
            app.config['RESULTS_FOLDER'],
            f"results_{timestamp}.csv"
        )
        results_df.to_csv(results_path, index=False)
        
        # Prepare response
        response = {
            'success': True,
            'summary': summary,
            'all_candidates': all_candidates,
            'top_candidates': top_candidates,
            'plot_url': f"/static/plots/{os.path.basename(plot_paths['combined'])}",
            'download_url': f"/static/results/{os.path.basename(results_path)}",
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'filename': filename or f"results_{timestamp}"
        }
        
        return jsonify(response)
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        app.logger.error(f"Error processing file: {str(e)}")
        return jsonify({
            'error': f'Error processing file: {str(e)}'
        }), 500


@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    model_info = get_model_info()
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_info['loaded'],
        'timestamp': datetime.now().isoformat(),
        'version': 'Celara-1.0'
    })


@app.route('/model-info')
def model_info_endpoint():
    """Get detailed model information"""
    return jsonify(get_model_info())


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    REST API endpoint for programmatic predictions
    
    POST body: JSON with array of observations
    [
        {
            "target_id": "KIC12345",
            "period": 10.5,
            "epoch": 54900.0,
            ...
        }
    ]
    """
    try:
        data = request.get_json()
        
        if not data or not isinstance(data, list):
            return jsonify({
                'error': 'Request body must be a JSON array of observations'
            }), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        validate_csv_columns(df)
        
        # Run predictions
        results_df = predict_batch(df)
        
        # Return results
        results = results_df.to_dict('records')
        return jsonify({
            'success': True,
            'count': len(results),
            'predictions': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({
        'error': 'File too large. Maximum size is 50MB.'
    }), 413


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    app.logger.error(f"Internal error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("Starting Celara...")
    model_info = get_model_info()
    
    if model_info['loaded']:
        print("Model loaded successfully")
        print(f"   Architecture: {model_info['architecture']}")
        if model_info['parameters']:
            print(f"   Parameters: {model_info['parameters']:,}")
    else:
        print("Running in demo mode (model not found)")
        print("   Place 'celara_model.keras' in the models/keras/ directory")
    
    print("Starting Flask server...")
    print("   Local: http://localhost:5000")
    print("   Network: http://0.0.0.0:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)