# AstroNet + Simple ResNet - Best of both worlds
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def simple_resnet_block(x, filters, kernel_size=5):
    """Simple 1D ResNet block - just the essentials"""
    # Main path
    shortcut = x
    x = layers.Conv1D(filters, kernel_size, padding='same', activation='relu')(x)
    x = layers.Conv1D(filters, kernel_size, padding='same')(x)
    
    # Skip connection (adjust channels if needed)
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, padding='same')(shortcut)
    
    # Add and activate
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

def create_astronet_resnet_trimodal():
    """
    Lightweight AstroNet + ResNet hybrid - Much more parameter efficient!
    
    - Uses Global Average Pooling instead of huge flattened layers
    - Aggressive downsampling to reduce feature maps
    - Smaller dense layers for classification
    - Should be ~500K parameters instead of 9M!
    """
    
    # ===== LOCAL VIEW (201 features - transit region) =====
    local_input = layers.Input(shape=(201,), name='local_view')
    local_x = layers.Reshape((201, 1))(local_input)
    
    # Initial conv with aggressive downsampling
    local_x = layers.Conv1D(16, 5, padding='same', activation='relu')(local_x)
    local_x = layers.MaxPooling1D(5, strides=5)(local_x)  # ~40 length
    
    # Simple ResNet blocks
    local_x = simple_resnet_block(local_x, 32)
    local_x = layers.MaxPooling1D(4, strides=4)(local_x)  # ~10 length
    
    # Global Average Pooling instead of flatten (HUGE parameter reduction!)
    local_features = layers.GlobalAveragePooling1D()(local_x)  # Just 32 features!
    
    # ===== GLOBAL VIEW (2001 features - full orbital period) =====
    global_input = layers.Input(shape=(2001,), name='global_view')
    global_x = layers.Reshape((2001, 1))(global_input)
    
    # Aggressive initial downsampling
    global_x = layers.Conv1D(16, 5, padding='same', activation='relu')(global_x)
    global_x = layers.MaxPooling1D(10, strides=10)(global_x)  # ~200 length
    
    # Progressive ResNet blocks with downsampling
    global_x = simple_resnet_block(global_x, 32)
    global_x = layers.MaxPooling1D(5, strides=5)(global_x)  # ~40 length
    
    global_x = simple_resnet_block(global_x, 64)
    global_x = layers.MaxPooling1D(4, strides=4)(global_x)  # ~10 length
    
    # Global Average Pooling (massive parameter reduction!)
    global_features = layers.GlobalAveragePooling1D()(global_x)  # Just 64 features!
    
    # ===== AUXILIARY FEATURES (4 stellar parameters) =====
    aux_input = layers.Input(shape=(4,), name='aux_features')
    aux_features = layers.Dense(8, activation='relu')(aux_input)  # Keep it simple
    
    # ===== CLASSIFICATION HEAD =====
    # Concatenate: 32 (local) + 64 (global) + 8 (aux) = 104 features total
    combined = layers.Concatenate()([local_features, global_features, aux_features])
    
    # Much smaller dense layers
    x = layers.Dense(128, activation='relu')(combined)  # 104 -> 128
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)          # 128 -> 64
    x = layers.Dropout(0.3)(x)
    
    # Output layer
    output = layers.Dense(1, activation='sigmoid', name='planet_probability')(x)
    
    # Create model with 3 inputs
    model = Model(
        inputs=[local_input, global_input, aux_input],
        outputs=output,
        name='AstroNet_ResNet_Lightweight'
    )
    
    return model