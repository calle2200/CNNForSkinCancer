import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_metadata(csv_path):
    return pd.read_csv(csv_path)

def load_images(df, image_dir, img_size=(224, 224), binary_classification=False):
    images = []
    labels = []

    if binary_classification:
        # 0 = benign, 1 = malignant
        cancer_labels = ['mel', 'bcc', 'akiec']
        df['label'] = df['dx'].apply(lambda dx: 1 if dx in cancer_labels else 0)
        label_map = {0: 0, 1: 1}
    else:
        label_map = {label: idx for idx, label in enumerate(df['dx'].unique())}
        df['label'] = df['dx'].map(label_map)

    for _, row in df.iterrows():
        img_path = os.path.join(image_dir, row['image_id'] + ".jpg")
        if not os.path.exists(img_path):
            continue
        image = load_img(img_path, target_size=img_size)
        image = img_to_array(image) / 255.0
        images.append(image)
        labels.append(row['label'])

    images = np.array(images)
    labels = tf.keras.utils.to_categorical(labels, num_classes=len(label_map))
    return images, labels, label_map
