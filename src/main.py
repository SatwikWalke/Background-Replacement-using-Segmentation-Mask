# main.py

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from utils import preprocess_image, get_foreground_mask
import config

def main():
    # Load and process image
    image_tensor = preprocess_image(config.INPUT_IMAGE_PATH)
    foreground_mask = get_foreground_mask(image_tensor)

    # Load background image and resize it to match the input image size
    background_image = Image.open(config.BACKGROUND_IMAGE_PATH)
    input_image = Image.open(config.INPUT_IMAGE_PATH)

    # Resize background to match input dimensions
    background_image = background_image.resize(input_image.size)

    # Convert to NumPy arrays
    input_image_np = np.array(input_image)
    background_image_np = np.array(background_image)

    # Blend images based on the segmentation mask
    output_image = np.where(foreground_mask[..., None], input_image_np, background_image_np)

    # Display final output
    plt.imshow(output_image)
    plt.title("Background Replaced Image")
    plt.axis('off')
    plt.show()

    # Save the final image
    output_image_pil = Image.fromarray(output_image)
    output_image_pil.save(config.OUTPUT_IMAGE_PATH)

if __name__ == '__main__':
    main()
