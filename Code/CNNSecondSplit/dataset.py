import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import random
import matplotlib.pyplot as plt

class CustomAudioDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_names = ["original", "deepfake"]
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []

        for dir_name in os.listdir(self.root_dir):
            dir_path = os.path.join(self.root_dir, dir_name)

            # Determine if the sample is "original" or "deepfake"
            label = 1 if dir_name.endswith("original") else 0

            samples.append((dir_path, label))

        return samples

    def __len__(self):
        return len(self.samples)



    def __getitem__(self, idx):
        try:
            dir_path, label = self.samples[idx]

            images = []

            for img_name in sorted(os.listdir(dir_path)):
                # Check if the image is from the right channel and frequency
                if "right-channel-frequency" in img_name:
                    img_path = os.path.join(dir_path, img_name)
                    img = Image.open(img_path).convert('RGB')

                    img = bottom_crop(img, height=230, width=1000)
                    if self.transform:
                        img = self.transform(img)

                    images.append(img)

            # Concatenate images along the channel dimension (dimension 0)
            sample = torch.cat(images, dim=0)
            ##plt.imshow(transforms.ToPILImage()(sample))
            ##plt.show()
            return sample, label

        except Exception as e:
            print(f"Error processing sample at index {idx}: {str(e)}")
            return None, None




def bottom_crop(image, height, width):
    # Assuming image is a PIL Image object
    img_width, img_height = image.size
    # Calculate the top left point of the crop area
    left = (img_width - width) / 2
    top = img_height - height
    right = (img_width + width) / 2
    bottom = img_height

    return image.crop((left, top, right, bottom))

'''
def bottom_crop(image, crop_height, width):
    img_width, img_height = image.size
    # Calculate the top left point of the crop area
    left = 0
    top = 170  # Skip the top 170 pixels
    right = img_width
    bottom = top + total_height  # Keep the next 1000 pixels in height

    return image.crop((left, top, right, bottom))
'''