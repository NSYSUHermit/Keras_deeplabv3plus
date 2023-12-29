from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import math

# define object mask loss
def custom_loss(y_true, y_pred):
    object_mask = tf.cast(y_true > 0, tf.float32)
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    loss = loss * object_mask[:,:,:,0]
    return tf.reduce_mean(loss)

# define loss (Dice Loss)
def dice_loss(y_true, y_pred, epsilon=1e-8):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2. * intersection + epsilon) / (union + epsilon)
    return 1 - dice

# cosineAnnealingLR
def lr_schedule_cos(epoch):
    initial_lr = 0.001
    T = 10 # cycle epoch
    M = initial_lr
    t = epoch % T
    lr = M * (1 + math.cos(t * math.pi / T)) / 2
    return lr

# backbone_dict = {'fast': , 'normal': }
def lr_schedule_warmup(epoch):
    initial_lr = 0.001
    decay_rate = 0.9
    warmup_epochs = 10
    warmup_lr = initial_lr / warmup_epochs
    if epoch < warmup_epochs:
        return float(warmup_lr * (epoch + 1))
    else:
        return float(initial_lr * decay_rate ** (epoch - warmup_epochs))

def convolution_block(
        block_input,
        num_filters=256,
        kernel_size=3,
        dilation_rate=1,
        padding="same",
        use_bias=False,
):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
    )(block_input)
    x = layers.BatchNormalization()(x)
    return tf.nn.relu(x)


def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)
    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)
    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output


def DeeplabV3Plus_Resnet(image_size, num_classes, mode='fast'):
    model_input = keras.Input(shape=(image_size, image_size, 3))
    resnet50 = keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=model_input
    )

    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
    return keras.Model(inputs=model_input, outputs=model_output)


def DeeplabV3Plus_Xception(image_size, num_classes):
    model_input = keras.Input(shape=(image_size, image_size, 3))

    Xception_model = tf.keras.applications.Xception(
        weights="imagenet", include_top=False, input_tensor=model_input)

    xception_x1 = Xception_model.get_layer("block12_sepconv3_act").output #(None, 32, 32, 728)：block9_sepconv3_act, block12_sepconv3_act, block13_sepconv3_act, block14_sepconv2
    x = DilatedSpatialPyramidPooling(xception_x1)

    input_a = layers.UpSampling2D(size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]), interpolation="bilinear")(x)
    input_a = layers.AveragePooling2D(pool_size=(2, 2))(input_a)
    xception_x2 = Xception_model.get_layer("block4_sepconv1_act").output
    input_b = convolution_block(xception_x2, num_filters=256, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(size=(image_size // x.shape[1], image_size // x.shape[2]), interpolation="bilinear")(x)
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
    return keras.Model(inputs=model_input, outputs=model_output)

if __name__ == '__main__':
    DeeplabV3Plus()
