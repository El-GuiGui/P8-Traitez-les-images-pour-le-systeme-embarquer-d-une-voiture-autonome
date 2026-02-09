from tensorflow.keras import layers, Model


def conv_block(x, filters):
    x = layers.Conv2D(
        filters, 3, padding="same", use_bias=False, kernel_initializer="he_normal"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(
        filters, 3, padding="same", use_bias=False, kernel_initializer="he_normal"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def unet_scratch(input_shape=(256, 256, 3), n_classes=8, base=32):
    inputs = layers.Input(shape=input_shape)

    c1 = conv_block(inputs, base)
    p1 = layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, base * 2)
    p2 = layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, base * 4)
    p3 = layers.MaxPooling2D()(c3)

    c4 = conv_block(p3, base * 8)
    p4 = layers.MaxPooling2D()(c4)

    bn = conv_block(p4, base * 16)

    u4 = layers.UpSampling2D()(bn)
    u4 = layers.Concatenate()([u4, c4])
    c5 = conv_block(u4, base * 8)

    u3 = layers.UpSampling2D()(c5)
    u3 = layers.Concatenate()([u3, c3])
    c6 = conv_block(u3, base * 4)

    u2 = layers.UpSampling2D()(c6)
    u2 = layers.Concatenate()([u2, c2])
    c7 = conv_block(u2, base * 2)

    u1 = layers.UpSampling2D()(c7)
    u1 = layers.Concatenate()([u1, c1])
    c8 = conv_block(u1, base)

    outputs = layers.Conv2D(n_classes, 1, padding="same", activation="softmax")(c8)
    return Model(inputs, outputs, name="unet_scratch")


import tensorflow as tf
from tensorflow.keras import layers, Model


def unet_vgg16(
    input_shape=(256, 256, 3), n_classes=8, encoder_weights="imagenet", trainable=False
):
    base = tf.keras.applications.VGG16(
        include_top=False, weights=encoder_weights, input_shape=input_shape
    )
    base.trainable = trainable

    s1 = base.get_layer("block1_conv2").output
    s2 = base.get_layer("block2_conv2").output
    s3 = base.get_layer("block3_conv3").output
    s4 = base.get_layer("block4_conv3").output
    b = base.get_layer("block5_conv3").output

    def up(x, skip, f):
        x = layers.UpSampling2D()(x)
        x = layers.Concatenate()([x, skip])
        x = conv_block(x, f)
        return x

    x = up(b, s4, 512)
    x = up(x, s3, 256)
    x = up(x, s2, 128)
    x = up(x, s1, 64)

    outputs = layers.Conv2D(n_classes, 1, activation="softmax", padding="same")(x)
    return Model(base.input, outputs, name="unet_vgg16")


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model


@tf.keras.utils.register_keras_serializable(package="proj8")
class ResNet50Preprocess(layers.Layer):
    def call(self, inputs):
        x = inputs * 255.0
        return tf.keras.applications.resnet50.preprocess_input(x)


def unet_resnet50(
    input_shape=(256, 256, 3),
    n_classes=8,
    encoder_weights="imagenet",
    trainable=False,
):
    inputs = layers.Input(shape=input_shape, name="image_rgb_01")

    x = ResNet50Preprocess(name="resnet50_preprocess")(inputs)

    base = tf.keras.applications.ResNet50(
        include_top=False,
        weights=encoder_weights,
        input_tensor=x,
    )
    base.trainable = trainable

    s1 = base.get_layer("conv1_relu").output  # /2
    s2 = base.get_layer("conv2_block3_out").output  # /4
    s3 = base.get_layer("conv3_block4_out").output  # /8
    s4 = base.get_layer("conv4_block6_out").output  # /16
    b = base.get_layer("conv5_block3_out").output  # /32

    def up(x, skip, f):
        x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
        x = layers.Concatenate()([x, skip])
        x = conv_block(x, f)
        return x

    x = up(b, s4, 512)  # /16
    x = up(x, s3, 256)  # /8
    x = up(x, s2, 128)  # /4
    x = up(x, s1, 64)  # /2

    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)

    outputs = layers.Conv2D(n_classes, 1, activation="softmax", padding="same")(x)
    return Model(inputs, outputs, name="unet_resnet50")
