from PIL import Image
import numpy as np

img1 = np.array(Image.open("image1.jpg"))
img2 = np.array(Image.open("image2.jpg"))

def analyze_image(img, name):
    print(f"Ảnh: {name}")
    print("Shape:", img.shape)
    print("Tensor hạng:", len(img.shape))
    print("-" * 30)

analyze_image(img1, "Ảnh 1")
analyze_image(img2, "Ảnh 2")
