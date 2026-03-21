import numpy as np
import matplotlib.cm as cm
from PIL import Image

def test_colorization():
    print("Creating dummy grayscale image...")
    # Create a simple grayscale gradient
    width, height = 100, 100
    x = np.linspace(0, 255, width)
    y = np.linspace(0, 255, height)
    xv, yv = np.meshgrid(x, y)
    gray_array = ((xv + yv) / 2).astype(np.uint8)
    
    gray_img = Image.fromarray(gray_array, mode='L')
    print("Grayscale image created.")

    print("Applying colormap 'jet'...")
    # Logic from DocAnh.py
    normalized_gray = gray_array / 255.0
    colormap = cm.get_cmap('jet')
    colored_rgba = colormap(normalized_gray)
    colored_uint8 = (colored_rgba * 255).astype(np.uint8)
    colored_img = Image.fromarray(colored_uint8)
    
    if colored_img.mode == 'RGBA':
        colored_img = colored_img.convert('RGB')
    
    output_path = "test_output_color.png"
    colored_img.save(output_path)
    print(f"Success! Saved colored image to {output_path}")

if __name__ == "__main__":
    test_colorization()
