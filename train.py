import os
import time
import numpy as np
import tensorflow as tf
from glob import glob

from utils.utils_data import data_generator, get_random_index

from models.deeplabv3plus import DeeplabV3Plus_Resnet, DeeplabV3Plus_Xception, lr_schedule_cos, lr_schedule_warmup, dice_loss, custom_loss

IMAGE_SIZE = 512
BATCH_SIZE = 2
Freeze_batch_size = 4
NUM_CLASSES = 1
NUM_CLASSES = NUM_CLASSES + 1
Freeze_Train = False
Init_lr = 1e-3
DATA_DIR = "D:/dataset/CrackForest_seg"
BACKBONE = 'Xception' # Resnet, Xception
EPOCHS = 5
Train_Ratio = 0.8
Seed = 1
AUGMENTATION = True
lr_strategy = lr_schedule_cos # lr_schedule_warmup, lr_schedule_cos
loss_strategy = None # dice_loss, custom_loss

image_list = sorted(glob(os.path.join(DATA_DIR, "Image/*")))
label_list = sorted(glob(os.path.join(DATA_DIR, "Label/*")))

train_index, val_index = get_random_index(image_list, ratio=Train_Ratio, seed=Seed)

train_images = [image_list[i]  for i in train_index]
train_masks = [label_list[i] for i in train_index]
val_images = [image_list[i] for i in val_index]
val_masks = [label_list[i] for i in val_index]

if BACKBONE == "Xception":
    model = DeeplabV3Plus_Xception(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)
elif BACKBONE == "Resnet":
    model = DeeplabV3Plus_Resnet(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)

model.summary()

if Freeze_Train:
    # ResNet: 138, Xception: 82
    if BACKBONE == "Xception":
        freeze_layers = 82
    elif BACKBONE == "ResNet":
        freeze_layers = 138

    for i in range(freeze_layers): model.layers[i].trainable = False
    print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model.layers)))

batch_size = Freeze_batch_size if Freeze_Train else BATCH_SIZE
train_dataset = data_generator(batch_size, train_images, train_masks, do_augment=AUGMENTATION, model=BACKBONE)
val_dataset = data_generator(batch_size, val_images, val_masks, model=BACKBONE)

print("Train Dataset:", train_dataset)
print("Val Dataset:", val_dataset)
dataset_size = tf.data.experimental.cardinality(train_dataset).numpy()
print(f"Train Dataset Size: {dataset_size}")


## Training process
if loss_strategy is None:
    loss_strategy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam() #SGD(momentum=0.9) #Adam(learning_rate=0.001)

model.compile(
    optimizer= optimizer,
    loss=loss_strategy,
    metrics=["accuracy"],
)

# define lr policy
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_strategy)

# define save best
checkpoint_path = 'best_model.h5'
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_best_only=True,
    monitor='val_loss',
    mode='min',
    verbose=1
)

# define early stop
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

s = time.time()
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[lr_scheduler, checkpoint_callback] # early_stopping
)
e = time.time()
print(f"training cost {int(e - s)} seconds!")

from matplotlib import pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

model.save(F'./weights/deeplabv3_{BACKBONE}_{DATA_DIR.split("/")[-1]}_{EPOCHS}_{int(e - s)}.h5')
