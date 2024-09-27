import numpy as np
import cv2
from skimage import io, color, transform, feature
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from scipy.stats import kurtosis, skew
import pandas as pd
from skimage.feature import graycomatrix, graycoprops


def extractColor(image_path):
    # print("hi")
    img = cv2.imread(image_path)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    skew_values = []
    kurtosis_values = []
    mean_values = []
    standard_values = []

    img_mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 0

    for color_channel in cv2.split(image):

        color_channel_data = color_channel[img_mask]

        kurtosis_values.append(kurtosis(color_channel_data))
        skew_values.append(skew(color_channel_data))
        mean_values.append(np.mean(color_channel_data))
        standard_values.append(np.std(color_channel_data))

    color_features = kurtosis_values + skew_values + mean_values + standard_values
    return color_features

# use the histogram to get the color distribution
# def colorHisto(image_name):
#     img = cv2.imread(image_name)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img_mask = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) > 0

#     hist_val = []
#     for i, color in enumerate(['r', 'g', 'b']):
#         hist_score = cv2.calcHist([img], [i], img_mask.astype(np.uint8), [256], [0, 256])
#         hist_val.extend(hist_score.flatten())
#     print(hist_val)

def haralick_fea(image_name):

    img = cv2.imread(image_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask = np.any(img != 0, axis=-1)
    dist = [1]
    angle = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    three_channel_features = []

    for i, color_channel in enumerate(['r', 'g', 'b']):
        # creates a greycomatrix 
        convert_image = (img[:, :, i] / img[:, :, i].max() * 255).astype(np.uint8)
        convert_image = convert_image * mask

        glcm_vals = graycomatrix(convert_image, dist, angle, 256, symmetric=True, normed=True)

        haralick_tex_fea = []
        for properties in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
            haralick_tex_fea.extend(graycoprops(glcm_vals, properties).flatten())
        # print(haralick_tex_fea)
        three_channel_features.append(haralick_tex_fea)

        features = [item for sublist in three_channel_features for item in sublist]

    return features

color_data = []
haralick_features = []
file_path = 'Skin_cancer_dataset/test/mask_nv_test/'
color_path = 'Features/test/color/color_nv.csv'
haralick_path = 'Features/test/color/haralick_nv.csv'
for filename in os.listdir(file_path):
    # print(filename)
    if os.path.splitext(filename)[1].lower() == '.jpg':
        image_path = os.path.join(file_path, filename)
        features = extractColor(image_path=image_path)
        color_data.append([[filename] + ['nv'] + ['benign'] +features])
        # print(image_path)
        # histogram method
        # histo_features = colorHisto(image_name=image_path)

        # haralick texture of colored region
        haralick_vals = haralick_fea(image_name=image_path)
        haralick_features.append([filename] + ['nv'] + ['benign'] + haralick_vals)


# haralick features in csv format
labels = ['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation', 'ASM']
columns = ['Image_name'] + ['Type'] + ['Mal-Ben'] + [f'{prop}_{color}_{angle}' for color in ['R', 'G', 'B'] for prop in labels for angle in range(1, 5)]
# print(len(columns))
haralick_dataframe = pd.DataFrame(haralick_features, columns=columns)
haralick_dataframe.to_csv(haralick_path, index=False)


# color related features like mean, skewness, kurtosis, deviation.
columns = ['Image_name', 'Type' ,'Mal-Ben', 'kurtosis_L', 'kurtosis_A', 'kurtosis_B', 'skew_L', 'skew_A', 'skew_B',
            'mean_L', 'mean_A', 'mean_B', 'std_L', 'std_A', 'std_B']
color_data = np.vstack(color_data)
color_dataframe = pd.DataFrame(color_data, columns=columns)
color_dataframe.to_csv(color_path, index = False)