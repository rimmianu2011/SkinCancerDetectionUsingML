import os
import shutil
import random
import pandas as pd


# to create the train, test and validate foldrs of images 
def randomly_select_and_move_images(source_folder, destination_folder, num_images):

    if not os.path.exists(source_folder):
        print(f"The source folder {source_folder} does not exist.")
        return
    

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    

    files = [file for file in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, file))]
    

    if len(files) < num_images:
        print(f"Requested {num_images} files, but only found {len(files)} files in the source folder.")
        return
    

    selected_files = random.sample(files, num_images)
    

    for file in selected_files:
        source_path = os.path.join(source_folder, file)
        destination_path = os.path.join(destination_folder, file)
        
    
        shutil.copy(source_path, destination_path)
        
        
        os.remove(source_path)
        
    print(f"Successfully moved {num_images} images to {destination_folder}.")


source_folder = 'Skin_cancer_dataset/train/mel/'
validate_folder = 'Skin_cancer_dataset/validate/mel_val'
test_folder = 'Skin_cancer_dataset/test/mel_test'
num_images = 215

# randomly_select_and_move_images(source_folder, test_folder, num_images)

# to combine all the features extracted so far for color and texture 
def combineCsv():
    directory_path = 'Features/test/asymmetry/'
    # directory_path = 'Features/test/color/'

    dataframes_list = []

    for filename in os.listdir(directory_path):
        
        if filename.startswith('asymmetry'):
        # if filename.startswith('haralick'):
        # if filename.startswith('color'):
            file_path = os.path.join(directory_path, filename)
            
            df = pd.read_csv(file_path)

            dataframes_list.append(df)

    combined_df = pd.concat(dataframes_list, ignore_index=True)

    combined_df = combined_df.drop_duplicates()

    combined_df.to_csv('Features/test/asymmetry/asymmetry.csv', index=False)
    # combined_df.to_csv('Features/test/color/color.csv', index=False)
    # combined_df.to_csv('Features/test/color/haralick.csv', index=False)


# combineCsv()

# to combine all the features into one csv file
def combineColorHaralick():
    # file_color = pd.read_csv('Features/test/color/color.csv')
    # file_haralick = pd.read_csv('Features/test/color/haralick.csv')
    file_color = pd.read_csv('Features/test/asymmetry/asymmetry.csv')
    file_haralick = pd.read_csv('Features/test/color/col_har.csv')

    merged_files = pd.merge(file_color, file_haralick, on = "Image_name", how = "inner")

    # merged_files.to_csv('Features/test/color/col_har.csv', index = False)
    merged_files.to_csv('Features/test/all_features.csv', index = False)
    # merged_files = merged_files.drop(['Type_y', 'Mal-Ben_y'], axis=1, inplace=True)


# combineColorHaralick()


def removeColumns():
    feature_file = pd.read_csv('Features/test/all_features.csv')

    feature_file.drop(['Type_y', 'Mal-Ben_y', 'Type', 'Mal-Ben'], axis=1, inplace=True)

    feature_file.to_csv('Features/test/Features_all.csv', index = False)

# removeColumns()


def moveColumn():
    file = pd.read_csv('Features/test/Features_all.csv')

    columnToMove = 'Mal-Ben_x'

    columns = [col for col in file.columns if col != columnToMove] + [columnToMove]

    file = file[columns]

    file.to_csv('Features/test/fea.csv', index = False)

# moveColumn()


def updateVal():
    pd.set_option('mode.chained_assignment', None)
    pd.set_option('future.no_silent_downcasting', True)

    file = pd.read_csv('Features/test/fea.csv')
    mappngVal = {'benign':0, 'malignant':1}

    file['Mal-Ben_x'] = file['Mal-Ben_x'].replace(mappngVal)

    file.to_csv('Features/test/test_features.csv', index=False)

# updateVal()


def arrangeCol():
    # Load the reference CSV file
    reference_df = pd.read_csv('Features/train/fea_all.csv')

    # Get the column order from the reference DataFrame
    column_order = reference_df.columns.tolist()

    # Load the CSV file whose columns need to be reordered
    target_df = pd.read_csv('Features/validate/validate_features.csv')

    # Reorder the columns of target_df to match the order in reference_df
    # If target_df contains columns not found in reference_df, they will be dropped
    # If reference_df contains columns not found in target_df, they will be ignored
    reordered_df = target_df[column_order]

    # Save the reordered DataFrame to a new CSV file
    reordered_df.to_csv('Features/validate/validate_features1.csv', index=False)


arrangeCol()