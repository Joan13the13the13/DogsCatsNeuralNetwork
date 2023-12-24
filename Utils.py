import cv2
import os

def get_min_dimensions(folder_path):
    min_width = float('inf')
    min_height = float('inf')

    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            try:
                img = cv2.imread(image_path)
                height, width, _ = img.shape
                min_width = min(min_width, width)
                min_height = min(min_height, height)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    return min_width, min_height