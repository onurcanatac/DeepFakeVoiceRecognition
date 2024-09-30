import shutil
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import soundfile as sf
from PIL import Image

def crop_image(file_path, left, top, right, bottom):
    with Image.open(file_path) as img:
        cropped_img = img.crop((left, top, img.width - right, img.height - bottom))
        cropped_img.save(file_path)

def generate_spectrograms(file_name, folder, save_dir):
    audio, sr = sf.read(os.path.join(folder, file_name), always_2d=True)
    file_base_name = os.path.splitext(file_name)[0]

    # Create a directory for each audio file
    file_dir = os.path.join(save_dir, file_base_name)
    os.makedirs(file_dir, exist_ok=True)

    segment_length = 5 * sr  # 5 seconds
    segment_counter = 0
    try:
        for start in range(0, audio.shape[0], segment_length):
            end = min(start + segment_length, audio.shape[0])
            segment_dir = os.path.join(file_dir, f'{segment_counter}. {start/sr}-{end/sr}seconds')
            os.makedirs(segment_dir, exist_ok=True)

            segment_left = audio[start:end, 0]   # Left channel
            segment_right = audio[start:end, 1]  # Right channel

            for channel, label in zip([segment_left, segment_right], ['left', 'right']):

                
            # Amplitude plot without margins
                amplitude_file_path = os.path.join(segment_dir, f'{label}-channel-amplitude.png')
                fig = plt.figure(figsize=(10, 4), frameon=False)
                ax = fig.add_axes([0, 0, 1, 1])  # Add axes that fill the figure
                ax.plot(np.linspace(start/sr, end/sr, num=len(channel)), channel)
                ax.axis('off')
                ax.set_position([0, 0, 1, 1])  # Set the axes' position to fill the figure
                plt.savefig(os.path.join(segment_dir, f'{label}-channel-amplitude.png'), bbox_inches='tight', pad_inches=0)
                plt.close()
                crop_image(amplitude_file_path, left=45, top=15, right=45, bottom=15)

                # Frequency plot (Spectrogram) without margins
                plt.figure(figsize=(10, 4), frameon=False)
                S = librosa.stft(channel)
                S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
                librosa.display.specshow(S_db, sr=sr)
                plt.axis('off')
                plt.gca().set_position([0, 0, 1, 1])
                plt.savefig(os.path.join(segment_dir, f'{label}-channel-frequency.png'), bbox_inches='tight', pad_inches=0)
                plt.close()

            segment_counter += 1
    except Exception as e:
        print(f"Error processing {file_name}: {e}")

# Specify the directories
audio_folders = {'FAKE': r'C:\Users\lenovo\OneDrive\Masaüstü\CS464ProjectLocal\KAGGLE\AUDIO\FAKE', 'REAL': r'C:\Users\lenovo\OneDrive\Masaüstü\CS464ProjectLocal\KAGGLE\AUDIO\REAL'}
save_dir = r'C:\Users\lenovo\OneDrive\Masaüstü\CS464ProjectLocal\Spectograms'

# Create spectrograms for each file
for label, folder in audio_folders.items():
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            generate_spectrograms(file, folder, save_dir)
