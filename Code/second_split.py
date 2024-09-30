# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 22:06:40 2023

@author: drbya
"""

""" Select two individual speakers and create three categories for the train test validation sets.
The categories will have:
Select train set for all files without person A and person B’s names in the title
Select the person A original and B to A (which are A-real and A-fake) voices.
Select the person B original and A to B (which are B-real and B-fake) voices.
"""
import os
import shutil
import random

# two speakers:
speaker_a = 'biden' # test
speaker_b = 'trump' # validation


# Function to copy files to train/validation/test folders with upper folder name
def copy_files(source_dir, dest_dir, piece_folders, upper_folder_name):
    os.makedirs(dest_dir, exist_ok=True)
    
    # the counter counts the number of 5 seconds intervals for original to fake ratios
    counter = [0,0]
    
    for folder in piece_folders:
        source_folder = os.path.join(source_dir, folder)
        dest_folder_name = f"{folder}_{upper_folder_name}"
        dest_folder = os.path.join(dest_dir, dest_folder_name)
        
        os.makedirs(dest_folder, exist_ok=True)  # Create destination folder

        if ("original" in upper_folder_name.lower()):
            counter[0] += 1
        else:
            counter[1] += 1
            
        # Iterate over files in the source folder and copy them to the destination folder
        for file in os.listdir(source_folder):
            source_path = os.path.join(source_folder, file)
            dest_filename = f"{file}"
            dest_path = os.path.join(dest_folder, dest_filename)

            # Perform the copy operation
            shutil.copy(source_path, dest_path)
    return counter

def add_arrays(count_train, count_train2):
    count_train[0] += count_train2[0]
    count_train[1] += count_train2[1]
    return count_train

# we have all files at a location
all_spectrogram_folders = r'C:\7. dönem\ml\project\spectograms2'
save_dir = r'C:\7. dönem\ml\project\split2'
subfolders = [f.path for f in os.scandir(all_spectrogram_folders) if f.is_dir()]

# we exclude the files that have neither speaker in their names for the train
# the counters [original,fake] count for each set
count_train = [0,0]
count_test = [0,0]
count_validation = [0,0]

for folder in subfolders:
    print(f"Processing folder: {folder}")
    upper_folder_name = os.path.basename(folder)
    files = os.listdir(folder)
    files = sorted(files, key=lambda x: int(x.split('.')[0]))  # Sort using numerical order
    
    if (not (speaker_a in folder.lower())) and (not(speaker_b in folder.lower())):
        count_train = add_arrays(count_train, copy_files(folder, os.path.join(save_dir, 'train'), files, upper_folder_name))

    # we then pick the test as ryan original and taylor to ryan
    elif ((f"{speaker_b}-to-{speaker_a}" in folder.lower()) or (f"{speaker_a}-original" in folder.lower())):
        count_test = add_arrays( count_test ,copy_files(folder, os.path.join(save_dir, 'test'), files, upper_folder_name))

    # we then pick the validation as taylor original and ryan to taylor
    elif ((f"{speaker_a}-to-{speaker_b}" in folder.lower()) or (f"{speaker_b}-original" in folder.lower())):
        count_validation  = add_arrays( count_validation, copy_files(folder, os.path.join(save_dir, 'validation'), files, upper_folder_name))
        
        
print(f"training had {count_train[0]} original data and {count_train[1]} fake data")
print(f"test had {count_test[0]} original data and {count_test[1]} fake data")
print(f"validation had {count_validation[0]} original data and {count_validation[1]} fake data")




print("---------Partitioning 50-50 (original-fake) to have no bias in data--------")
def random_subset(save_dir, dest_dir, piece_folders, upper_folder_name, count):
    os.makedirs(dest_dir, exist_ok=True)
    
    original_files = [os.path.basename(file) for file in piece_folders if "original" in os.path.basename(file)]
    fake_files = [os.path.basename(file) for file in piece_folders if ("original" not in os.path.basename(file))]
    selected_files = random.sample(original_files, count) + random.sample(fake_files, count)
    counter = [0,0]
    for folder in selected_files:
        source_folder = os.path.join( save_dir, folder)
        dest_folder_name = f"{folder}"
        dest_folder = os.path.join(  os.path.join(dest_dir, upper_folder_name) , dest_folder_name)
        os.makedirs(dest_folder, exist_ok=True)  # Create destination folder
        if ("original" in dest_folder_name.lower()):
            counter[0] += 1
        else:
            counter[1] += 1
        # Iterate over files in the source folder and copy them to the destination folder
        for file in os.listdir(source_folder):
            dest_filename = f"{file}"
            source_path = os.path.join(source_folder, dest_filename)
            dest_path = os.path.join(dest_folder, dest_filename)
            # Perform the copy operation
            shutil.copy(source_path, dest_path)
    return counter



# Calculate the minimum count among train, test, and validation sets
# By doing so, we will take equal number of data, preventing any bias
min_count_train = min(count_train[0],count_train[1])
min_count_test = min(count_test[0],count_test[1])
min_count_validation = min(count_validation[0],count_validation[1])

count_train = [0,0]
count_test = [0,0]
count_validation = [0,0]

# Create a folder for the random subset
random_subset_folder = os.path.join(save_dir, 'equal_distribution_data')
os.makedirs(random_subset_folder, exist_ok=True)

save_dir_train = os.path.join(save_dir, 'train')
save_dir_test = os.path.join(save_dir, 'test')
save_dir_validation = os.path.join(save_dir, 'validation')

subfolders_train = [f.path for f in os.scandir( save_dir_train) if f.is_dir()]
subfolders_test = [f.path for f in os.scandir( save_dir_test) if f.is_dir()]
subfolders_validation = [f.path for f in os.scandir( save_dir_validation) if f.is_dir()]

# Create random subsets for train, test, and validation
count_train = random_subset(save_dir_train, random_subset_folder, subfolders_train, "train", min_count_train)
print(f"Training now has {count_train[0]} original data and {count_train[1]} fake data")
count_test = random_subset(save_dir_test, random_subset_folder, subfolders_test, "test", min_count_test)
print(f"Test now has {count_test[0]} original data and {count_test[1]} fake data")
count_validation = random_subset(save_dir_validation, random_subset_folder, subfolders_validation, "validation", min_count_validation)
print(f"Validation now has {count_validation[0]} original data and {count_validation[1]} fake data")

