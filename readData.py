import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt 
import numpy as np
import os
import shutil

def readData():
    #read the metadata file to get the id of the images
    HAM_metadata = pd.read_csv('Skin_cancer_dataset/HAM10000_metadata.csv')
    size_img = (128, 128)
    array_images = []
    original_path = 'Skin_cancer_dataset/Ham10000_images_part_1/'
    destination_path = 'Skin_cancer_dataset/bkl/'
    for index, row in HAM_metadata.iterrows():
        # print(row['image_id'])
        image_orig = f"Skin_cancer_dataset/HAM10000_images_part_1/{row['image_id']}.jpg"
        # print(image_orig)
        # img_open = Image.open(image_orig)

        # #image processing for RGB
        # img_open = resize_image(img_open, size_img)
        # img_open = normalize_image(img_open)

        # img_array = np.array(img_open)
        # array_images.append(img_array.flatten())
        
        if row['dx'] == 'bkl':
            # print(row['dx'])
            type_skin = row['dx']
            image_name = row['image_id'] + '.jpg'

            source = os.path.join(original_path, image_name)
            # target = os.path.join(destination_path, type_skin)

            if not os.path.exists(destination_path):
                os.makedirs(destination_path)

            shutil.copy(source, os.path.join(destination_path, image_name))

    # col = [f'pixel_{i}' for i in range(128*128*3)]
    # image_data_frame = pd.DataFrame(array_images, columns=col)
    # image_data_frame.to_csv('Skin_cancer_dataset/HAM_128_RGB.csv', index = False)

def normalize_image(img):
    normal_im = np.array(img)
    # print('2')
    normal_im = normal_im.astype('float32') / 255.0
    return normal_im

def resize_image(img, new_size):
    width, height = img.size
    # print('hello')
    # print(width, height)
    new_img = img.resize(new_size)
    return new_img
    


readData()







'''.DS_Store
import cv2
import numpy as np
from skimage import measure, morphology
from matplotlib import pyplot as plt

def load_image(image_path):
    return cv2.imread(image_path)

def segment_lesion(image):
    # Convert to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Thresholding to get the segmentation mask
    _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations to clean up noise
    kernel = np.ones((5,5),np.uint8)
    closing = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
    
    return closing

def extract_asymmetry_features(segmented_image):
    # Label the segmented image
    labeled_img = measure.label(segmented_image)
    props = measure.regionprops(labeled_img)
    
    if not props:
        print("No regions found")
        return None
    
    # Assuming the largest region is the lesion
    lesion_region = max(props, key=lambda x: x.area)
    
    # Calculate asymmetry features
    original_area = lesion_region.area
    original_perimeter = lesion_region.perimeter
    
    # Flip the segmented image horizontally
    flipped_image = np.fliplr(segmented_image)
    labeled_flipped_img = measure.label(flipped_image)
    props_flipped = measure.regionprops(labeled_flipped_img)
    
    flipped_region = max(props_flipped, key=lambda x: x.area)
    
    flipped_area = flipped_region.area
    flipped_perimeter = flipped_region.perimeter
    
    # Calculate asymmetry
    area_asymmetry = abs(original_area - flipped_area)
    perimeter_asymmetry = abs(original_perimeter - flipped_perimeter)
    
    return area_asymmetry, perimeter_asymmetry

# Example usage
image_path = 'path/to/your/image.jpg' # Update this to the path of your image
image = load_image(image_path)
segmented = segment_lesion(image)
area_asymmetry, perimeter_asymmetry = extract_asymmetry_features(segmented)

print(f"Area Asymmetry: {area_asymmetry}, Perimeter Asymmetry: {perimeter_asymmetry}")


# columns = ['image_filename', 'skin_cancer_type'] + [f'feature_{i}' for i in range(len(asymmetry_features[0]) - 2)]
# asym_df = pd.DataFrame(asymmetry_features, columns=columns)

'''



'''.DS_Store












# def estimate_vignetting(img):
#     img_re = cv2.resize(img, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
#     rows, cols = img_re.shape[:2]
#     # Create coordinate grid
#     x, y = np.meshgrid(np.arange(cols), np.arange(rows))
#     # Create an array with the intensities of the pixels
#     z = img_re.flatten()
#     # Fit radial basis function to the intensity data
#     rbf = Rbf(x.flatten(), y.flatten(), z, function='multiquadric', smooth=0.0)
#     # Evaluate the function on the grid
#     z_fit = rbf(x, y)
#     z_fit = cv2.resize(z_fit.reshape(rows, cols), (img.shape[1], img.shape[0]))
#     vignetting_pattern = z_fit / np.max(z_fit)  # Normalize the pattern
#     return vignetting_pattern

# def correct_vignetting(image):
#     # Load the image
#     img = cv2.imread(image)
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     # Estimate the vignetting pattern from the grayscale image
#     vignetting_pattern = estimate_vignetting(img_gray)

#     # Correct the original image
#     img_float = img.astype(np.float32)
#     corrected_img = img_float / vignetting_pattern[:,:,np.newaxis]

#     # Clip values to the correct range and convert to uint8
#     corrected_img= np.clip(corrected_img, 0, 255).astype(np.uint8)

#     cv2.imshow('Corrected Image', corrected_img)
#     cv2.waitKey(1000)

#     return corrected_img

'''