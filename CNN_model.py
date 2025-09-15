import tensorflow as tf

def encoder_block(inputs, num_filters):
    x = tf.keras.layers.Conv2D(num_filters, 3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(num_filters, 3, padding='same', activation='relu')(x)
    p = tf.keras.layers.MaxPooling2D((2,2))(x)
    return x, p

def decoder_block(inputs, skip_features, num_filters):
    x = tf.keras.layers.Conv2DTranspose(num_filters, (2,2), strides=2, padding='same')(inputs)
    x = tf.keras.layers.Concatenate()([x, skip_features])
    x = tf.keras.layers.Conv2D(num_filters, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(num_filters, 3, padding='same', activation='relu')(x)
    return x

def build_unet(input_shape=(256,256,3), num_classes=1):
    inputs = tf.keras.layers.Input(input_shape)

    s1, p1 = encoder_block(inputs, 32)
    s2, p2 = encoder_block(p1, 64)
    s3, p3 = encoder_block(p2, 128)

    b1 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(p3)
    b1 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(b1)

    d1 = decoder_block(b1, s3, 128)
    d2 = decoder_block(d1, s2, 64)
    d3 = decoder_block(d2, s1, 32)

    outputs = tf.keras.layers.Conv2D(num_classes, 1, padding='same', activation='sigmoid')(d3)

    model = tf.keras.models.Model(inputs, outputs, name="Compact_U-Net")
    return model

if __name__ == "__main__":
    model = build_unet()
    model.summary()
