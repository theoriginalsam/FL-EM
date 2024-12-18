import tensorflow as tf
from tensorflow.keras import layers, Model
from config.settings import MODEL_CONFIG

def conv_block(x, filters, dropout_rate=0.3):
    """
    Create a convolutional block with batch normalization and dropout
    """
    x = layers.Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def build_unet(input_shape=(64, 64, 8)):
    """
    Build U-Net model for deforestation detection
    """
    # Input
    inputs = layers.Input(input_shape)
    
    # Encoder
    enc1 = conv_block(inputs, 32, dropout_rate=0.1)
    pool1 = layers.MaxPooling2D()(enc1)
    
    enc2 = conv_block(pool1, 64, dropout_rate=0.1)
    pool2 = layers.MaxPooling2D()(enc2)
    
    enc3 = conv_block(pool2, 128, dropout_rate=0.2)
    pool3 = layers.MaxPooling2D()(enc3)
    
    # Bridge
    bridge = conv_block(pool3, 256, dropout_rate=0.2)
    
    # Decoder
    up1 = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(bridge)
    merge1 = layers.Concatenate()([enc3, up1])
    dec1 = conv_block(merge1, 128, dropout_rate=0.2)
    
    up2 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(dec1)
    merge2 = layers.Concatenate()([enc2, up2])
    dec2 = conv_block(merge2, 64, dropout_rate=0.1)
    
    up3 = layers.Conv2DTranspose(32, 2, strides=2, padding='same')(dec2)
    merge3 = layers.Concatenate()([enc1, up3])
    dec3 = conv_block(merge3, 32, dropout_rate=0.1)
    
    # Output
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(dec3)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

def compile_model(model, learning_rate=MODEL_CONFIG['LEARNING_RATE']):
    """
    Compile model with custom loss and metrics
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(threshold=0.5),
            tf.keras.metrics.Precision(thresholds=0.5),
            tf.keras.metrics.Recall(thresholds=0.5),
            tf.keras.metrics.AUC()
        ]
    )
    return model

def get_callbacks(model_name='best_model.keras'):
    """
    Get training callbacks
    """
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            model_name,
            save_best_weights_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]