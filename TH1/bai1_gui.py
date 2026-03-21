import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import os

class ImageAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Analysis Tool")
        self.root.geometry("600x600")

        # Title Label
        self.label_title = tk.Label(root, text="Chương trình phân tích ảnh", font=("Arial", 16, "bold"))
        self.label_title.pack(pady=10)

        # Select Button
        self.btn_select = tk.Button(root, text="Chọn ảnh", command=self.load_image, font=("Arial", 12))
        self.btn_select.pack(pady=5)

        # Image Display Area
        self.lbl_image = tk.Label(root)
        self.lbl_image.pack(pady=10)

        # Info Display Area
        self.lbl_info = tk.Label(root, text="", font=("Arial", 12), justify="left")
        self.lbl_info.pack(pady=10, padx=20, fill="x")

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Chọn file ảnh",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )

        if not file_path:
            return

        try:
            # Open and process image
            image = Image.open(file_path)
            
            # Display image (resize for thumbnail)
            display_img = image.copy()
            display_img.thumbnail((400, 400))
            self.tk_image = ImageTk.PhotoImage(display_img)
            self.lbl_image.config(image=self.tk_image)

            # Analyze with numpy
            img_array = np.array(image)
            name = os.path.basename(file_path)
            
            shape = img_array.shape
            rank = len(shape) # Tensor rank is the number of dimensions

            result_text = (
                f"Tệp: {name}\n"
                f"Kích thước (Shape): {shape}\n"
                f"Hạng Tensor (Rank): {rank}"
            )
            self.lbl_info.config(text=result_text)

        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể xử lý ảnh: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageAnalyzerApp(root)
    root.mainloop()
