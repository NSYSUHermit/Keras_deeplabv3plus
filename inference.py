import os
import cv2
import random
from glob import glob
import tensorflow as tf
from models.deeplabv3plus import dice_loss
from utils.utils_data import read_image, infer, get_random_index, compute_dir_miou

Train_Ratio = 0.8
Seed = 1
DATA_DIR = r"D:/dataset/CrackForest_seg" # D:/dataset/Hulk_Exp_Dataset/DONE_1.3/Magnetic_seg ;/CrackForest_seg; /MVTec_Metal_seg
num_classes = 1
image_list = sorted(glob(os.path.join(DATA_DIR, "Image/*")))
label_list = sorted(glob(os.path.join(DATA_DIR, "Label/*")))

train_index, val_index = get_random_index(image_list, ratio=Train_Ratio, seed=Seed)

train_images1 = [image_list[i] for i in train_index]
train_masks = [label_list[i] for i in train_index]
val_images = [image_list[i] for i in val_index]
val_masks = [label_list[i] for i in val_index]

model = tf.keras.models.load_model('./best_model.h5') #./weights/deeplabv3_xception_crack_200_3130.h5, custom_objects={'dice_loss': dice_loss}, best_model.h5

output_folder = val_images[0].split('\\')[0] + "/keras_output/"
os.makedirs(output_folder, exist_ok=True)

# calculate miou
miou = compute_dir_miou(num_classes, val_images, val_masks, model)
print(miou)

# save predict image
for image_file in val_images:
    image = cv2.imread(image_file)
    image = cv2.resize(image, (512, 512))
    image_tensor, image_shape = read_image(image_file)
    prediction_mask = infer(image_tensor=image_tensor, model=model)
    for i in range(prediction_mask.max()):
        random.seed(i+1)
        image[prediction_mask == i+1] = (random.randint(0,200), random.randint(0,200), random.randint(0,200))
    output_name = output_folder +  image_file.split('\\')[2]
    cv2.imwrite(output_name, image)
