import numpy as np
import cv2
from skimage import io, color, transform, feature, filters, measure
import os
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from PIL import Image, ImageFilter
from skimage.segmentation import chan_vese

def hairRemoval(image_name, image, inpaint_file):
    orig_img = cv2.imread(image_name)

    # orig_img = image

    # color -> grayScale
    img_gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    
    # morphological operation to highlight the dark hair on the skin
    kernel_size = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))  
    blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel_size)
    """ cv2.imshow("blackhat Image" ,blackhat)
    cv2.waitKey(800) """

    # thresholding to get the binary mask, this will get the hair from the image
    thresh_img = cv2.adaptiveThreshold(blackhat, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 45, -10)
    """ cv2.imshow("thresh Image" ,thresh_img)
    cv2.waitKey(800) """

    # Used dilation technique to make the hair more pronounced
    kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    remove_noise = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel_dil, iterations=2)
    dilate_hair = cv2.dilate(remove_noise, kernel_dil, iterations=4)
    # cv2.imshow("dilate Image" ,dilate_hair)
    # cv2.waitKey(800)

    # fills in the hair region in the original image
    inpaint_val = 11
    img_inpaint = cv2.inpaint(orig_img, dilate_hair, inpaint_val, cv2.INPAINT_TELEA)
    # cv2.imshow("inpaint Image" ,img_inpaint)
    # cv2.waitKey(800)
    cv2.imwrite(os.path.join(inpaint_file, os.path.basename(image_name)), img_inpaint)

    return img_inpaint


# Apply otsu's thresholding method followed by Chan-vese algo for segmented image.
def otsuChan(in_img, bin_file, image_name):
    # cv2.imshow("orig Image" ,in_img)
    # cv2.waitKey(200)

    # remove the noise from grayscale image
    img_gray = cv2.cvtColor(in_img, cv2.COLOR_RGB2GRAY)

    # img_YCrCb = cv2.cvtColor(in_img, cv2.COLOR_RGB2YCrCb)
    # lower_bound = np.array([100, 30, 75], dtype=np.uint8)
    # upper_bound = np.array([255, 220, 190], dtype=np.uint8)

    # img_HSV = cv2.cvtColor(in_img, cv2.COLOR_RGB2HSV)
    # lower_bound = np.array([10, 80, 120], dtype=np.uint8)  
    # upper_bound = np.array([230, 255, 255], dtype=np.uint8)

    # img_mask = cv2.inRange(img_YCrCb, lower_bound, upper_bound)

    # new_img = cv2.bitwise_and(in_img, in_img, mask=img_mask)
    # cv2.imshow("yc Image" , img_mask)
    # cv2.waitKey(200)
    # img_blurred = cv2.GaussianBlur(img_gray, (7, 7), 0)

    img_blurred = cv2.medianBlur(img_gray, 7)
    # img_blurred = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2GRAY)
    # applies otsu's thresholding technique to get the segmented image
    ret, binary_img = cv2.threshold(img_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, (7, 7))

    # to store the segmented images in a folder for further computations
    # cv2.imwrite(os.path.join(bin_file, "noise_" + os.path.basename(image)), img_blurred)
    # cv2.imwrite(os.path.join(bin_file, "binary_" + os.path.basename(image_name)), binary_img)

    # chan vese implementation
    chan_val = chan_vese(binary_img, lambda1=1, lambda2=1, tol=1e-3, max_num_iter=200, dt=0.5,
               init_level_set="checkerboard", extended_output=True)

    segmented_img = chan_val[0].astype(float)
    segmented_img = cv2.morphologyEx(segmented_img, cv2.MORPH_CLOSE, (45, 45))
    # cv2.imshow("chan Image" ,segmented_img)
    # cv2.waitKey(200)

    cv2.imwrite(os.path.join(bin_file, os.path.basename(image_name)), binary_img)
    # cv2.destroyAllWindows()

    return binary_img


# for getting the skin lesion using masking
def masking(bin_image, in_image, image_path, image_name):
    bin_image = cv2.bitwise_not(bin_image)
    new_img = cv2.bitwise_and(in_image, in_image, mask=bin_image)
    # cv2.imshow("masked Image" ,new_img)
    # cv2.waitKey(200)
    cv2.imwrite(os.path.join(image_path, os.path.basename(image_name)), new_img)



# for creating a mask on the segmented image
def segmentImage(image):
    # print('hi')
    # image = (image * 255).astype(np.uint8)
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = cv2.GaussianBlur(image, (1, 1), 0)
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    edges_img = cv2.Canny(closing, 60, 80)

    # Apply Sobel operator
    sobelx = cv2.Sobel(edges_img, cv2.CV_64F, 1, 0, ksize=5) # Sobel X
    sobely = cv2.Sobel(edges_img, cv2.CV_64F, 0, 1, ksize=5) # Sobel Y

    # Combine the two gradients
    sobel_combined = np.sqrt(sobelx**2 + sobely**2)
    
    # cv2.imshow("Segmented image" ,sobel_combined)
    # cv2.waitKey(1000)
    # cv2.destroyAllWindows()

# METHOD I for brightening the image
def brighten_image(image):
    orig_img = io.imread(image)

    rows, cols = orig_img.shape[:2]
    # Create a radial gradient mask
    orig_img_float = orig_img.astype(np.float32) / 255.0

    X = np.linspace(-1, 1, cols)
    Y = np.linspace(-1, 1, rows)
    X, Y = np.meshgrid(X, Y)
    radial_gradient = np.sqrt(X**2 + Y**2)
    radial_gradient = np.clip(radial_gradient, 0, 1)
    mask = 1 - radial_gradient * 0.55 + (1 - 0.55)  # Adjust 0.75 based on vignetting severity
    
    mask3 = cv2.merge([mask, mask, mask])  # Merge to make it 3-channel
    mask3 = mask3.astype(np.float32)

    # Apply the mask
    corrected_img = cv2.multiply(orig_img_float, mask3)
    corrected_img = np.clip(corrected_img, 0, 1) 
    corrected_img = (corrected_img * 255.0).astype(np.uint8)

    # Save or display the result
    """ cv2.imshow('Corrected Image', corrected_img)
    cv2.waitKey(1000)
 """
    return corrected_img


# METHOD II (converting to HSV) for brightening the image
def darkToBright(image):
    img = cv2.imread(image)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)

    new_v = np.clip(v + (50*(255-v)/255), 0, 255)

    merge_hsv = cv2.merge((h, s, new_v.astype(np.uint8)))
    
    new_img = cv2.cvtColor(merge_hsv, cv2.COLOR_HSV2BGR)

    # cv2.imshow("bright", new_img)
    # cv2.waitKey(1000)

    return new_img


# METHOD III (converting to LAB) for brightening the image

def bright_LAB(image):
    img = cv2.imread(image)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = img_lab[:, :, 0]

    val, thresh = cv2.threshold(L, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    val += 20 
    thresh = cv2.threshold(L, val, 255, cv2.THRESH_BINARY)[1]

    # invert threshold and make 3 channels
    thresh = 255 - thresh
    thresh = cv2.merge([thresh, thresh, thresh])

    gain = 3
    blue = cv2.multiply(img[:,:,0], gain)
    green = cv2.multiply(img[:,:,1], gain)
    red = cv2.multiply(img[:,:,2], gain)
    img_bright = cv2.merge([blue, green, red])

    # blend original and brightened using thresh as mask
    result = np.where(thresh==255, img_bright, img)

    cv2.imshow("bright", result)
    cv2.waitKey(1000)

    return result


file_path = 'Skin_cancer_dataset/test/nv_test/'
save_bin = 'Skin_cancer_dataset/test/binary_nv_test/'
inpaint_folder = 'Skin_cancer_dataset/test/inpaint_img_nv_test/'
masked_folder = 'Skin_cancer_dataset/test/mask_nv_test/'
index = 0
for filename in os.listdir(file_path):
    if os.path.splitext(filename)[1].lower() == '.jpg':
        image_path = os.path.join(file_path, filename)
        # img = darkToBright(image=image_path)
        # img = brighten_image(image=image_path)
        # img = bright_LAB(image=image_path)

        img_pro = hairRemoval(image_name=image_path, image = image_path, inpaint_file=inpaint_folder) 
        img_seg = otsuChan(in_img=img_pro, bin_file=save_bin, image_name=image_path)
        img_mask = masking(bin_image=img_seg, in_image=img_pro, image_path=masked_folder, image_name=image_path)
        # segmentImage(image=image_path)
        """ index = index + 1
        if (index == 25):
            break """