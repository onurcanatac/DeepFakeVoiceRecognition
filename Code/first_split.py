import os
import shutil
from PIL import Image

# Split data into train/validation/test sets
def split_data(data_dir, train_ratio=0.8, val_ratio=0.1):
    files = os.listdir(data_dir)

    files = sorted(files, key=lambda x: int(x.split('.')[0]))  # Sort using numerical order

    num_files = len(files)
    num_train = int(train_ratio * num_files)
    num_val = int(val_ratio * num_files)
    
    train_files = files[:num_train]
    val_files = files[num_train:num_train+num_val]
    test_files = files[num_train+num_val:]
    
    return train_files, val_files, test_files


# Copy files to train/validation/test folders, upper_folder_name is biden_original etc
def copy_files(source_dir, dest_dir, piece_folders, upper_folder_name):
    os.makedirs(dest_dir, exist_ok=True)

    # folder->5 second folders // file-> the files in the 5 second folder
    for folder in piece_folders:
        source_folder = os.path.join(source_dir, folder)
        dest_folder_name = f"{folder}_{upper_folder_name}"
        dest_folder = os.path.join(dest_dir, dest_folder_name)
        
        os.makedirs(dest_folder, exist_ok=True) 

        # Iterate over files in the 5-second folders and copy them to the destination folder
        for file in os.listdir(source_folder):
            source_path = os.path.join(source_folder, file)
            dest_filename = f"{file}"
            dest_path = os.path.join(dest_folder, dest_filename)

            # Perform the copy operation
            shutil.copy(source_path, dest_path)
    combine_images(dest_folder)

# Combine images into one collective image
def combine_images(dest_folder):
    combined_image = None

    for file in os.listdir(dest_folder):
        file_path = os.path.join(dest_folder, file)
        
        # Check if the file is an image (you might need to adjust the condition based on your image file format)
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            img = Image.open(file_path)

            if combined_image is None:
                combined_image = img
            else:
                combined_image = Image.new('RGB', (combined_image.width + img.width, max(combined_image.height, img.height)))
                combined_image.paste(combined_image, (0, 0))
                combined_image.paste(img, (combined_image.width, 0))

    if combined_image is not None:
        print(f"Combined image is saved to {dest_folder}")
        combined_image.save(os.path.join(dest_folder, 'combined_image.png'))

# Paths
all_spectrogram_folders = r'C:\onur_bilkent\CS464ProjectLocal\Spectograms'
save_dir = r'C:\onur_bilkent\CS464ProjectLocal\First Split'

# Get a list of subdirectories in all_spectogram_folders
subfolders = [f.path for f in os.scandir(all_spectrogram_folders) if f.is_dir()]

# Loop through subdirectories
for folder in subfolders:
    print(f"Processing folder: {folder}")

    # Get the list of files and split them
    train_files, val_files, test_files = split_data(folder)

    # Copy the files to train/validation/test folders with upper folder name
    upper_folder_name = os.path.basename(folder)

    copy_files(folder, os.path.join(save_dir, 'train'), train_files, upper_folder_name)
    copy_files(folder, os.path.join(save_dir, 'validation'), val_files, upper_folder_name)
    copy_files(folder, os.path.join(save_dir, 'test'), test_files, upper_folder_name)