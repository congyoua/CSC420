import cv2
import numpy as np
# This tool is used to search the BGR paletter which ground truths are using.
def extract_bgr_palette(image_path):
    # Load the image segmentation annotation
    img = cv2.imread(image_path)

    # Reshape the array
    img_reshaped = img.reshape(-1, 3)

    # Find the unique colors
    unique_colors = np.unique(img_reshaped, axis=0)

    return unique_colors

if __name__ == "__main__":
    # Set the path to the image segmentation annotation
    image_path = "../data/labels/0001_clothes.png"

    # Extract the RGB palette
    bgr_palette = extract_bgr_palette(image_path)

    # Print the RGB palette
    print(bgr_palette)

    # 1 [ 0   0 128]
    # 2 [  0 128   0]
    # 3 [0, 128, 128]
    # 4 [128   0   0]
    # 5 [128   0 128]
    # 6 [128, 128 ,0]