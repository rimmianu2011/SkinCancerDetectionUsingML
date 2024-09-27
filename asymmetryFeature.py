import pandas as pd
from skimage import io, color, transform, filters, measure
import matplotlib.pyplot as plt 
import numpy as np
import os
import cv2
from scipy import ndimage 



def calAsymmVal(image_path):
    binary_img = io.imread(image_path)

    # rgb_gray = color.rgb2gray(orig_img)

    # threshold_val = filters.threshold_otsu(rgb_gray)

    # binary_img = rgb_gray > threshold_val

    # center_img = np.array(binary_img.shape) / 2

    # distance = np.sqrt(np.sum(np.indices(binary_img.shape) **2, axis=0))

    # edge_dis = distance*binary_img

    # mean_img = np.mean(edge_dis[edge_dis>0])

    # asym_score = np.mean(edge_dis - mean_img)

    label_img = measure.label(binary_img)
    props = measure.regionprops(label_img)[0]  # Assuming single object for simplicity

    # Solidity = Area / Convex Area
    solidity = props.solidity
    
    # Eccentricity: Measure of the elongation of the region
    eccentricity = props.eccentricity

    # Boundary Analysis - Perimeter irregularity (Perimeter^2 / Area)
    perimeter_irregularity = props.perimeter ** 2 / props.area

    return solidity, eccentricity, perimeter_irregularity

    # return asym_score

    # plt.imshow(binary_img)
    # plt.show()



HAM_metadata = pd.read_csv('Skin_cancer_dataset/HAM10000_metadata.csv')
file_path = 'Skin_cancer_dataset/test/binary_nv_test/'
image_folder = 'Skin_cancer_dataset/Ham10000_images_part_1/'
feature_folder = 'Features/test/asymmetry/'
asym_feature = []

# for index, row in HAM_metadata.iterrows():
for filename in os.listdir(file_path):
    # print(filename)
    if os.path.splitext(filename)[1].lower() == '.jpg':
        image_path = os.path.join(file_path, filename)

        # image_name = row['image_id'] + '.jpg'
        # image_path = os.path.join(image_folder, f"{image_name}")
        # asymmetry = calAsymmVal(image_path)
        solidity_score, eccentricity_score, perimeter_irregularity_score = calAsymmVal(image_path)
        img_id = filename
        
        asym_lab = np.array([[img_id, 'nv', 'benign', solidity_score, eccentricity_score, perimeter_irregularity_score]])
        asym_feature.append(asym_lab)

    # index = index + 1
    # if (index == 10):
    #     break


asym_feature = np.vstack(asym_feature)
columns = ['Image_name', 'Type', 'Mal-Ben', 'solidity', 'eccentricity', 'perimeter']
asym_df = pd.DataFrame(asym_feature, columns=columns)

asym_df.to_csv(feature_folder + "asymmetry_nv.csv", index=False)