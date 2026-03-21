import sys
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np

# Redirect stdout to handle unicode if needed
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

class ImageConsoleApp:
    def __init__(self, file_path):
        self.file_path = file_path
        self.original_img = None
        self.img_array = None
        self.rank = 0
        self.shape = ()

    def load_image(self):
        try:
            # 4. Hiển thị bức ảnh gốc (Load ảnh)
            self.original_img = Image.open(self.file_path)
            
            # Chuyển đổi sang mảng NumPy để xử lý như Tensor
            self.img_array = np.array(self.original_img)
            self.rank = self.img_array.ndim
            self.shape = self.img_array.shape
            return True
        except FileNotFoundError:
            print(f"Lỗi: Không tìm thấy tệp tin tại {self.file_path}")
            return False
        except Exception as e:
            print(f"Đã xảy ra lỗi: {e}")
            return False

    def print_info(self):
        # 2. Hiển thị thông tin kích thước và Tensor
        print("--- THÔNG TIN ẢNH ---")
        print(f"Bức ảnh là Tensor hạng: {self.rank}")
        print(f"Giá trị các chiều (Shape): {self.shape}")

    def print_matrix(self):
        # 3. Hiển thị giá trị ma trận điểm ảnh
        print("\n--- MA TRẬN ĐIỂM ẢNH ---")
        if self.rank == 2:
            print("Ảnh đen trắng (Grayscale):")
            print(self.img_array)
        else:
            print(f"Ảnh màu với {self.shape[2]} kênh (RGB/RGBA):")
            for i in range(self.shape[2]):
                print(f"Kênh màu {i}:")
                # Handle potential issue if image has no channels (unlikely if rank > 2 but safe to slice)
                print(self.img_array[:, :, i])

    def show_gui(self):
        # Khởi tạo giao diện Tkinter
        root = tk.Tk()
        root.title("Chương trình Xử lý Ảnh (OOP)")
        
        # Layout chính
        main_frame = tk.Frame(root, padx=10, pady=10)
        main_frame.pack()

        def add_image_to_ui(img_obj, title):
            # Hàm hỗ trợ hiển thị ảnh lên UI
            frame = tk.LabelFrame(main_frame, text=title)
            frame.pack(side=tk.LEFT, padx=5)
            
            # Resize ảnh để hiển thị vừa màn hình nếu quá lớn
            display_size = (300, 300)
            img_copy = img_obj.copy()
            img_copy.thumbnail(display_size)
            
            img_tk = ImageTk.PhotoImage(img_copy)
            label = tk.Label(frame, image=img_tk)
            label.image = img_tk # Giữ reference
            label.pack()

        # Hiển thị ảnh gốc
        if self.original_img:
            add_image_to_ui(self.original_img, "Ảnh Gốc")

            # 5. Nếu là ảnh màu, chuyển thành xám và hiển thị
            if self.rank > 2:
                gray_img = self.original_img.convert('L')
                add_image_to_ui(gray_img, "Ảnh Xám")
            
            # 6. Thực hiện phép chuyển vị (Transpose)
            if self.rank == 3:
                transposed_array = np.transpose(self.img_array, (1, 0, 2))
            else:
                transposed_array = self.img_array.T
                
            transposed_img = Image.fromarray(transposed_array)
            add_image_to_ui(transposed_img, "Ảnh Chuyển Vị")

        # 7. Đóng chương trình
        btn_close = tk.Button(root, text="Đóng chương trình", command=root.destroy, bg="#ff4444", fg="white")
        btn_close.pack(pady=10)

        root.mainloop()

    def run(self):
        if self.load_image():
            self.print_info()
            self.print_matrix()
            self.show_gui()

if __name__ == "__main__":
    # 1. Nhận tham số đường dẫn từ console
    if len(sys.argv) > 1:
        path = sys.argv[1]
        app = ImageConsoleApp(path)
        app.run()
    else:
        # Fallback default path
        default_path = "d:/HMUD-K22/Image/anh-meme-1.jpg"
        print(f"Không có tham số dòng lệnh. Đang dùng ảnh mặc định: {default_path}")
        app = ImageConsoleApp(default_path)
        app.run()
