import pandas as pd
from skimage import io, color, transform, filters
from skimage.feature import graycomatrix, graycoprops
import matplotlib.pyplot as plt 
import numpy as np
import os


HAM_metadata = pd.read_csv('Skin_cancer_dataset/HAM10000_metadata.csv')
image_folder = 'Skin_cancer_dataset/Ham10000_images_part_1/'
feature_folder = 'Features/'
asymmetry_features = []
asym_feature = []

for index, row in HAM_metadata.iterrows():
    image_name = row['image_id'] + '.jpg'
    image_path = os.path.join(image_folder, f"{image_name}")


    index = index + 1
    if (index == 10):
        break