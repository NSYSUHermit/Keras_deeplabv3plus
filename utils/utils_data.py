from imgaug import augmenters as iaa
import imgaug as ia
import tensorflow as tf
import numpy as np
import cv2

img_size = 512
IMAGE_SIZE = img_size

def read_image(image_path, mask=False, model="Xception"):
    image = tf.io.read_file(image_path)
    w, h = 0, 0
    if mask:
        image = tf.image.decode_png(image, channels=1)
        image.set_shape([None, None, 1])
        image = tf.image.resize(images=image, size=[img_size, img_size])
    else:
        image = tf.image.decode_png(image, channels=3)
        image.set_shape([None, None, 3])
        h, w, _ = image.shape
        image = tf.image.resize(images=image, size=[img_size, img_size])
        if model == "Xception":
            image = tf.keras.applications.xception.preprocess_input(image)
        elif model == "Resnet":
            image = tf.keras.applications.resnet50.preprocess_input(image)

    return image, [w, h]

def load_data(image_list, mask_list, model="Xception"):
    image, _ = read_image(image_list, model)
    mask, _ = read_image(mask_list, mask=True, model=model)
    return image, mask

rand_aug = iaa.RandAugment(n=3, m=7)
AUTO = tf.data.AUTOTUNE

def augment(images):
    # Input to `augment()` is a TensorFlow tensor which
    # is not supported by `imgaug`. This is why we first
    # convert it to its `numpy` variant.
    images = tf.cast(images, tf.uint8)
    return rand_aug(images=images.numpy())

def data_augmentation(image, mask):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)

    image = tf.image.random_brightness(image, max_delta=0.2)

    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    return image, mask

def data_generator(batch_size, image_list, mask_list, do_augment=False, model="Xception"):
    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE, model=model)
    if do_augment:
        dataset = dataset.map(data_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset

def infer(model, image_tensor):
    predictions = model.predict(np.expand_dims((image_tensor), axis=0))
    predictions = np.squeeze(predictions)
    predictions = np.argmax(predictions, axis=2)
    return predictions

def get_random_index(image_list, ratio=0.8, seed=1):
    dataset_size = len(image_list)
    train_size = int(dataset_size * ratio)
    data_index = set(range(dataset_size))
    np.random.seed(seed)
    train_index = set(np.random.choice(a=dataset_size, size=train_size, replace=False))
    val_index = data_index.difference(train_index)
    return train_index, val_index

def compute_miou(num_classes, y_pred, y_true):
    iou_per_class = []
    for i in range(1, num_classes + 1):
        pred = y_pred == i
        true = y_true[:, :, 0] == i
        intersection = np.logical_and(true, pred)
        union = np.logical_or(true, pred)

        if np.sum(union) != 0:
            iou = np.sum(intersection) / np.sum(union)
            iou_per_class.append(iou)

    miou = np.mean(iou_per_class) if iou_per_class else 1
    return miou

def compute_dir_miou(num_classes, val_images, val_masks, model):
    miou_per_image = []
    for i in range(len(val_images)):
        image_tensor, image_shape = read_image(val_images[i])
        vaL_label = cv2.resize(cv2.imread(val_masks[i]), (512, 512))
        prediction_mask = infer(image_tensor=image_tensor, model=model)
        miou = compute_miou(num_classes, prediction_mask, vaL_label)
        miou_per_image.append(miou)
    dir_miou = np.mean(miou_per_image)
    return dir_miou
