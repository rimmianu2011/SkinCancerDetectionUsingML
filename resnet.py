import pandas as pd
from skimage import io, color, transform, filters
from skimage.feature import graycomatrix, graycoprops
import matplotlib.pyplot as plt 
import numpy as np
import os
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image



resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(600,
               450, 3))

def fea_extract(image_path):
    orig_img = image.load_img(image_path, target_size=(600, 450))
    img_con = image.img_to_array(orig_img)
    img_exp = np.expand_dims(img_con, axis=0)
    img_preprocess = preprocess_input(img_exp)
    img_feature = resnet_model.predict(img_preprocess)

    return img_feature.flatten() 


HAM_metadata = pd.read_csv('Skin_cancer_dataset/HAM10000_metadata.csv')
image_folder = 'Skin_cancer_dataset/Ham10000_images_part_1/'
feature_folder = 'Features/'
asymmetry_features = []
# asym_feature = []

for index, row in HAM_metadata.iterrows():
    image_name = row['image_id'] + '.jpg'
    image_path = os.path.join(image_folder, f"{image_name}")
    asymmetry = fea_extract(image_path)
    img_id = row['image_id']
    type_skin = row['dx']
    asym_lab = np.concatenate(([img_id, type_skin], asymmetry))
    # asym_lab = np.array([[img_id, type_skin, asymmetry]])
    asymmetry_features.append(asym_lab)
    # asym_feature.append(asym_lab)

    index = index + 1
    if (index == 10):
        break


asym_feature = np.vstack(asym_feature)
columns = ['image_filename', 'skin_cancer_type', 'feature_score']
asym_df = pd.DataFrame(asym_feature, columns=columns)

asym_df.to_csv(feature_folder + "resnet50_asym.csv", index=False)