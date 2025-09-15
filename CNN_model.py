import tensorflow as tf

def encoder_block(inputs, num_filters, dropout_rate=0.3):
    """Encoder block with higher dropout for small dataset"""
    x = tf.keras.layers.Conv2D(num_filters, 3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(num_filters, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    p = tf.keras.layers.MaxPooling2D((2,2))(x)
    return x, p

def decoder_block(inputs, skip_features, num_filters, dropout_rate=0.3):
    """Decoder block with higher dropout for small dataset"""
    x = tf.keras.layers.Conv2DTranspose(num_filters, (2,2), strides=2, padding='same')(inputs)
    x = tf.keras.layers.Concatenate()([x, skip_features])
    x = tf.keras.layers.Conv2D(num_filters, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(num_filters, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x

def dice_loss(y_true, y_pred, smooth=1e-6):
    """Dice loss function for segmentation tasks"""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Dice coefficient metric"""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def build_unet(input_shape=(256,256,3), num_classes=1, dropout_rate=0.1):
    inputs = tf.keras.layers.Input(input_shape)

    # Encoder
    s1, p1 = encoder_block(inputs, 32, dropout_rate)
    s2, p2 = encoder_block(p1, 64, dropout_rate)
    s3, p3 = encoder_block(p2, 128, dropout_rate)
    s4, p4 = encoder_block(p3, 256, dropout_rate)

    # Bridge
    b1 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')(p4)
    b1 = tf.keras.layers.BatchNormalization()(b1)
    b1 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')(b1)
    b1 = tf.keras.layers.BatchNormalization()(b1)
    b1 = tf.keras.layers.Dropout(dropout_rate)(b1)

    # Decoder
    d1 = decoder_block(b1, s4, 256, dropout_rate)
    d2 = decoder_block(d1, s3, 128, dropout_rate)
    d3 = decoder_block(d2, s2, 64, dropout_rate)
    d4 = decoder_block(d3, s1, 32, dropout_rate)

    outputs = tf.keras.layers.Conv2D(num_classes, 1, padding='same', activation='sigmoid')(d4)

    model = tf.keras.models.Model(inputs, outputs, name="Improved_U-Net")
    return model

if __name__ == "__main__":
    model = build_unet()
    model.summary()