import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model, optimizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def simple_resnet_block(x, filters, kernel_size=3, dropout=0.1):
    """Simple ResNet block with batch norm and dropout"""
    shortcut = x
    
    x = layers.Conv1D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout)(x)
    
    x = layers.Conv1D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

def create_simple_resnet_trimodal():
    """ResNet CNN for exoplanet detection - targeting ~2M parameters"""
    
    # LOCAL VIEW - More depth and wider channels
    local_input = layers.Input(shape=(201,), name='local_view')
    local_x = layers.Reshape((201, 1))(local_input)
    
    local_x = layers.Conv1D(64, 5, padding='same', activation='relu')(local_x)
    local_x = layers.BatchNormalization()(local_x)
    local_x = layers.MaxPooling1D(2)(local_x)
    
    # Multiple ResNet blocks with increasing depth
    local_x = simple_resnet_block(local_x, 64, kernel_size=5, dropout=0.1)
    local_x = simple_resnet_block(local_x, 64, kernel_size=5, dropout=0.1)
    local_x = layers.MaxPooling1D(2)(local_x)
    
    local_x = simple_resnet_block(local_x, 128, kernel_size=3, dropout=0.1)
    local_x = simple_resnet_block(local_x, 128, kernel_size=3, dropout=0.1)
    local_x = layers.MaxPooling1D(2)(local_x)
    
    local_x = simple_resnet_block(local_x, 256, kernel_size=3, dropout=0.2)
    local_x = simple_resnet_block(local_x, 256, kernel_size=3, dropout=0.2)
    local_x = layers.GlobalAveragePooling1D()(local_x)
    
    # GLOBAL VIEW - More depth and wider channels
    global_input = layers.Input(shape=(2001,), name='global_view')
    global_x = layers.Reshape((2001, 1))(global_input)
    
    global_x = layers.Conv1D(64, 7, padding='same', activation='relu')(global_x)
    global_x = layers.BatchNormalization()(global_x)
    global_x = layers.MaxPooling1D(5)(global_x)
    
    global_x = simple_resnet_block(global_x, 64, kernel_size=5, dropout=0.1)
    global_x = simple_resnet_block(global_x, 64, kernel_size=5, dropout=0.1)
    global_x = layers.MaxPooling1D(4)(global_x)
    
    global_x = simple_resnet_block(global_x, 128, kernel_size=3, dropout=0.1)
    global_x = simple_resnet_block(global_x, 128, kernel_size=3, dropout=0.1)
    global_x = layers.MaxPooling1D(4)(global_x)
    
    global_x = simple_resnet_block(global_x, 256, kernel_size=3, dropout=0.2)
    global_x = simple_resnet_block(global_x, 256, kernel_size=3, dropout=0.2)
    global_x = layers.GlobalAveragePooling1D()(global_x)
    
    # AUXILIARY FEATURES - Expanded network
    aux_input = layers.Input(shape=(4,), name='aux_features')
    aux_x = layers.Dense(64, activation='relu')(aux_input)
    aux_x = layers.BatchNormalization()(aux_x)
    aux_x = layers.Dense(128, activation='relu')(aux_x)
    aux_x = layers.BatchNormalization()(aux_x)
    aux_features = layers.Dense(256, activation='relu')(aux_x)
    
    # FEATURE FUSION - Larger combined feature space
    # 256 (local) + 256 (global) + 256 (aux) = 768 features
    combined = layers.Concatenate()([local_x, global_x, aux_features])
    
    # CLASSIFICATION HEAD - Deeper and wider
    x = layers.Dense(512, activation='relu')(combined)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    output = layers.Dense(1, activation='sigmoid', name='exoplanet_probability')(x)
    
    model = Model(
        inputs=[local_input, global_input, aux_input],
        outputs=output,
        name='ResNet_Exoplanet_2M'
    )
    
    return model

def get_simple_callbacks(model_path='models/keras/simple_resnet_model.keras', 
                         patience=15, factor=0.5, min_lr=1e-6):
    """Simple callbacks for training"""
    callback_list = [
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=patience,
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1,
            save_weights_only=False
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=factor,
            patience=patience//3,
            min_lr=min_lr,
            mode='max',
            verbose=1
        )
    ]
    
    return callback_list