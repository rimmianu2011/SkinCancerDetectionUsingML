import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt 
import numpy as np
import os
import shutil

def bklCreate():
    #read the metadata file to get the id of the images
    HAM_metadata = pd.read_csv('Skin_cancer_dataset/HAM10000_metadata.csv')
    original_path = 'Skin_cancer_dataset/Ham10000_images_part_1/'
    destination_path1 = 'Skin_cancer_dataset/bkl/'

    for index, row in HAM_metadata.iterrows():
        image_orig = f"Skin_cancer_dataset/HAM10000_images_part_1/{row['image_id']}.jpg"
        
        if row['dx'] == 'bkl':
            type_skin = row['dx']
            image_name = row['image_id'] + '.jpg'

            source = os.path.join(original_path, image_name)

            if not os.path.exists(destination_path1):
                os.makedirs(destination_path1)

            shutil.copy(source, os.path.join(destination_path1, image_name))

def dfCreate():
    #read the metadata file to get the id of the images
    HAM_metadata = pd.read_csv('Skin_cancer_dataset/HAM10000_metadata.csv')
    original_path = 'Skin_cancer_dataset/Ham10000_images_part_1/'
    destination_path3 = 'Skin_cancer_dataset/df/'

    for index, row in HAM_metadata.iterrows():
        image_orig = f"Skin_cancer_dataset/HAM10000_images_part_1/{row['image_id']}.jpg"
        
        if row['dx'] == 'df':
            type_skin = row['dx']
            image_name = row['image_id'] + '.jpg'

            source = os.path.join(original_path, image_name)

            if not os.path.exists(destination_path3):
                os.makedirs(destination_path3)

            shutil.copy(source, os.path.join(destination_path3, image_name))

def melCreate():
    #read the metadata file to get the id of the images
    HAM_metadata = pd.read_csv('Skin_cancer_dataset/HAM10000_metadata.csv')
    original_path = 'Skin_cancer_dataset/Ham10000_images_part_1/'
    destination_path2 = 'Skin_cancer_dataset/mel/'

    for index, row in HAM_metadata.iterrows():
        image_orig = f"Skin_cancer_dataset/HAM10000_images_part_1/{row['image_id']}.jpg"
        
        if row['dx'] == 'mel':
            type_skin = row['dx']
            image_name = row['image_id'] + '.jpg'

            source = os.path.join(original_path, image_name)

            if not os.path.exists(destination_path2):
                os.makedirs(destination_path2)

            shutil.copy(source, os.path.join(destination_path2, image_name))

def vascCreate():
    #read the metadata file to get the id of the images
    HAM_metadata = pd.read_csv('Skin_cancer_dataset/HAM10000_metadata.csv')
    original_path = 'Skin_cancer_dataset/Ham10000_images_part_1/'
    destination_path2 = 'Skin_cancer_dataset/vasc/'

    for index, row in HAM_metadata.iterrows():
        image_orig = f"Skin_cancer_dataset/HAM10000_images_part_1/{row['image_id']}.jpg"
        
        if row['dx'] == 'vasc':
            type_skin = row['dx']
            image_name = row['image_id'] + '.jpg'

            source = os.path.join(original_path, image_name)

            if not os.path.exists(destination_path2):
                os.makedirs(destination_path2)

            shutil.copy(source, os.path.join(destination_path2, image_name))

def bccCreate():
    #read the metadata file to get the id of the images
    HAM_metadata = pd.read_csv('Skin_cancer_dataset/HAM10000_metadata.csv')
    original_path = 'Skin_cancer_dataset/Ham10000_images_part_1/'
    destination_path2 = 'Skin_cancer_dataset/bcc/'

    for index, row in HAM_metadata.iterrows():
        image_orig = f"Skin_cancer_dataset/HAM10000_images_part_1/{row['image_id']}.jpg"
        
        if row['dx'] == 'bcc':
            type_skin = row['dx']
            image_name = row['image_id'] + '.jpg'

            source = os.path.join(original_path, image_name)

            if not os.path.exists(destination_path2):
                os.makedirs(destination_path2)

            shutil.copy(source, os.path.join(destination_path2, image_name))

def nvCreate():
    #read the metadata file to get the id of the images
    HAM_metadata = pd.read_csv('Skin_cancer_dataset/HAM10000_metadata.csv')
    original_path = 'Skin_cancer_dataset/Ham10000_images_part_1/'
    destination_path2 = 'Skin_cancer_dataset/nv/'

    for index, row in HAM_metadata.iterrows():
        image_orig = f"Skin_cancer_dataset/HAM10000_images_part_1/{row['image_id']}.jpg"
        
        if row['dx'] == 'nv':
            type_skin = row['dx']
            image_name = row['image_id'] + '.jpg'

            source = os.path.join(original_path, image_name)

            if not os.path.exists(destination_path2):
                os.makedirs(destination_path2)

            shutil.copy(source, os.path.join(destination_path2, image_name))

def akiecCreate():
    #read the metadata file to get the id of the images
    HAM_metadata = pd.read_csv('Skin_cancer_dataset/HAM10000_metadata.csv')
    original_path = 'Skin_cancer_dataset/Ham10000_images_part_1/'
    destination_path2 = 'Skin_cancer_dataset/akiec/'

    for index, row in HAM_metadata.iterrows():
        image_orig = f"Skin_cancer_dataset/HAM10000_images_part_1/{row['image_id']}.jpg"
        
        if row['dx'] == 'akiec':
            type_skin = row['dx']
            image_name = row['image_id'] + '.jpg'

            source = os.path.join(original_path, image_name)

            if not os.path.exists(destination_path2):
                os.makedirs(destination_path2)

            shutil.copy(source, os.path.join(destination_path2, image_name))

dfCreate()
melCreate()
bklCreate()
vascCreate()
bccCreate()
nvCreate()
akiecCreate()