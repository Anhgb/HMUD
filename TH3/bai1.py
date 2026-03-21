import sys
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np

def process_image(file_path):
    try:
        # 4. Hiển thị bức ảnh gốc (Load ảnh)
        original_img = Image.open(file_path)
        
        # Chuyển đổi sang mảng NumPy để xử lý như Tensor
        img_array = np.array(original_img)
        
        # 2. Hiển thị thông tin kích thước và Tensor
        print("--- THÔNG TIN ẢNH ---")
        rank = img_array.ndim
        shape = img_array.shape
        print(f"Bức ảnh là Tensor hạng: {rank}")
        print(f"Giá trị các chiều (Shape): {shape}")
        
        # 3. Hiển thị giá trị ma trận điểm ảnh
        print("\n--- MA TRẬN ĐIỂM ẢNH ---")
        if rank == 2:
            print("Ảnh đen trắng (Grayscale):")
            print(img_array)
        else:
            print(f"Ảnh màu với {shape[2]} kênh (RGB/RGBA):")
            for i in range(shape[2]):
                print(f"Kênh màu {i}:")
                print(img_array[:, :, i])
        
        # Khởi tạo giao diện Tkinter
        root = tk.Tk()
        root.title("Chương trình Xử lý Ảnh")
        
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
            label.image = img_tk # Giữ reference để không bị garbage collected
            label.pack()

        # Hiển thị ảnh gốc
        add_image_to_ui(original_img, "Ảnh Gốc")

        # 5. Nếu là ảnh màu, chuyển thành xám và hiển thị
        if rank > 2:
            gray_img = original_img.convert('L')
            add_image_to_ui(gray_img, "Ảnh Xám")
        
        # 6. Thực hiện phép chuyển vị (Transpose)
        # Đối với ma trận 3D (H, W, C), ta chuyển vị H và W
        if rank == 3:
            transposed_array = np.transpose(img_array, (1, 0, 2))
        else:
            transposed_array = img_array.T
            
        transposed_img = Image.fromarray(transposed_array)
        add_image_to_ui(transposed_img, "Ảnh Chuyển Vị")

        # 7. Đóng chương trình
        btn_close = tk.Button(root, text="Đóng chương trình", command=root.destroy, bg="#ff4444", fg="white")
        btn_close.pack(pady=10)

        root.mainloop()

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy tệp tin tại {file_path}")
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")

if __name__ == "__main__":
    # 1. Nhận tham số đường dẫn từ console
    if len(sys.argv) > 1:
        path = sys.argv[1]
        process_image(path)
    else:
        # Fallback default path if no argument is provided
        default_path = "d:/HMUD-K22/Image/image2.jpg"
        print(f"Không có tham số dòng lệnh. Đang dùng ảnh mặc định: {default_path}")
        process_image(default_path)