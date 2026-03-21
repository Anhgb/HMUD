import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from PIL import Image, ImageTk
import numpy as np
import sys
import os

# Redirect stdout to handle unicode if needed
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bài 2: Giao diện Đồ họa (Modern UI)")
        self.root.geometry("1100x700")
        self.root.configure(bg="#f3f4f6") # Light gray background

        # Data
        self.original_image = None # PIL Image
        self.current_image = None  # PIL Image (currently displayed)
        self.image_array = None    # Numpy Array

        self.setup_styles()
        self.setup_ui()
        self.setup_menu()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')

        # Colors
        bg_color = "#f3f4f6"
        accent_color = "#2563eb" # Blue
        text_color = "#1f2937"
        header_bg = "#ffffff"
        
        # Styles
        style.configure("TLabel", background=header_bg, foreground=text_color, font=("Segoe UI", 10))
        style.configure("Card.TFrame", background=header_bg, relief="flat")
        style.configure("TButton", 
                        font=("Segoe UI", 10, "bold"), 
                        background=accent_color, 
                        foreground="white", 
                        borderwidth=0, 
                        focuscolor=accent_color)
        style.map("TButton", background=[("active", "#1d4ed8")])
        style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"), foreground="#111827", background=header_bg)
        style.configure("SubHeader.TLabel", font=("Segoe UI", 11, "bold"), foreground="#374151", background=header_bg)

    def setup_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # 1. File Menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Hệ thống", menu=file_menu)
        file_menu.add_command(label="Mở ảnh", command=self.open_image)
        file_menu.add_separator()
        file_menu.add_command(label="Thoát", command=self.root.quit)

        # 2. Process Menu
        process_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Xử lý", menu=process_menu)
        process_menu.add_command(label="Chuyển sang ảnh xám", command=self.convert_to_grayscale)
        process_menu.add_command(label="Chuyển vị ảnh", command=self.transpose_image)
        process_menu.add_command(label="Reset về ảnh gốc", command=self.reset_image)

        # 3. View Menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Hiển thị", menu=view_menu)
        view_menu.add_command(label="Thông tin tensor & Ma trận", command=self.show_image_info)


    def setup_ui(self):
        # Main Container
        main_container = ttk.Frame(self.root, style="Card.TFrame")
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Header
        header_frame = ttk.Frame(main_container, style="Card.TFrame")
        header_frame.pack(fill=tk.X, padx=20, pady=(20, 10))
        ttk.Label(header_frame, text="BÀI 2: CHƯƠNG TRÌNH XỬ LÝ ẢNH", style="Header.TLabel").pack(side=tk.LEFT)

        # Toolbar Buttons
        toolbar_frame = ttk.Frame(main_container, style="Card.TFrame")
        toolbar_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        ttk.Button(toolbar_frame, text="📂 Mở Ảnh", command=self.open_image).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(toolbar_frame, text="🔄 Reset", command=self.reset_image).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(toolbar_frame, text="🌑 Grayscale", command=self.convert_to_grayscale).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(toolbar_frame, text="📐 Chuyển Vị", command=self.transpose_image).pack(side=tk.LEFT)

        # Split Layout
        self.paned_window = tk.PanedWindow(main_container, orient=tk.HORIZONTAL, bg="#e5e7eb", sashwidth=4, sashrelief="flat")
        self.paned_window.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

        # LEFT PANE: Image
        self.frame_left = tk.Frame(self.paned_window, bg="white")
        self.paned_window.add(self.frame_left)
        
        self.lbl_image = tk.Label(self.frame_left, text="Vui lòng mở ảnh...", bg="white", fg="#9ca3af", font=("Segoe UI", 12))
        self.lbl_image.pack(expand=True, fill=tk.BOTH, padx=2, pady=2)

        # RIGHT PANE: Info
        self.frame_right = tk.Frame(self.paned_window, bg="white")
        self.paned_window.add(self.frame_right)

        right_content = ttk.Frame(self.frame_right, style="Card.TFrame", padding=15)
        right_content.pack(fill=tk.BOTH, expand=True)

        ttk.Label(right_content, text="THÔNG TIN ẢNH & MA TRẬN", style="SubHeader.TLabel").pack(anchor="w", pady=(0, 10))
        
        self.txt_info = scrolledtext.ScrolledText(right_content, wrap=tk.NONE, font=("Consolas", 10), state='disabled', relief="flat", bg="#f9fafb", padx=10, pady=10)
        self.txt_info.pack(fill=tk.BOTH, expand=True)
        
        # Styles for text widget
        self.txt_info.tag_config("bold", font=("Consolas", 10, "bold"), foreground="#2563eb")
        self.txt_info.tag_config("dim", foreground="#6b7280")

    def open_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif"), ("All files", "*.*")]
        )
        if file_path:
            try:
                self.original_image = Image.open(file_path)
                self.reset_image()
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể mở ảnh: {e}")

    def display_image(self, pil_img):
        if not pil_img: return

        # Dynamic resize
        container_w = self.frame_left.winfo_width()
        container_h = self.frame_left.winfo_height()
        if container_w < 10: container_w = 600
        if container_h < 10: container_h = 500
        
        target_size = (container_w - 20, container_h - 20)
        
        img_copy = pil_img.copy()
        img_copy.thumbnail(target_size, Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.ANTIALIAS)
        
        self.tk_image = ImageTk.PhotoImage(img_copy)
        self.lbl_image.config(image=self.tk_image, text="")
        self.lbl_image.image = self.tk_image

    def reset_image(self):
        if self.original_image:
            self.current_image = self.original_image.copy()
            self.image_array = np.array(self.original_image)
            self.display_image(self.current_image)
            self.show_image_info()

    def convert_to_grayscale(self):
        if self.current_image:
            self.current_image = self.current_image.convert('L')
            self.image_array = np.array(self.current_image)
            self.display_image(self.current_image)
            self.show_image_info()
        else:
            messagebox.showwarning("Cảnh báo", "Chưa có ảnh nào được mở!")

    def transpose_image(self):
        if self.image_array is not None:
            rank = self.image_array.ndim
            if rank == 3:
                transposed_array = np.transpose(self.image_array, (1, 0, 2))
            else:
                transposed_array = self.image_array.T
            
            self.image_array = transposed_array
            
            try:
                self.current_image = Image.fromarray(self.image_array)
                self.display_image(self.current_image)
                self.show_image_info()
            except Exception as e:
                messagebox.showerror("Lỗi hiển thị", f"Không thể hiển thị ma trận chuyển vị: {e}")
        else:
            messagebox.showwarning("Cảnh báo", "Chưa có ảnh nào được mở!")

    def show_image_info(self):
        if self.image_array is None: return

        self.txt_info.config(state='normal')
        self.txt_info.delete(1.0, tk.END)
        
        rank = self.image_array.ndim
        shape = self.image_array.shape
        dtype = self.image_array.dtype
        
        self.txt_info.insert(tk.END, "THÔNG TIN CHI TIẾT\n", "bold")
        self.txt_info.insert(tk.END, f"• Rank (Số chiều): {rank}\n")
        self.txt_info.insert(tk.END, f"• Shape (Kích thước): {shape}\n")
        self.txt_info.insert(tk.END, f"• Kiểu dữ liệu: {dtype}\n\n")
        
        self.txt_info.insert(tk.END, "MA TRẬN ĐIỂM ẢNH (Snippet 20x20)\n", "bold")
        
        limit = 20
        
        if rank == 2:
            self.txt_info.insert(tk.END, "[Ảnh Đen Trắng]\n", "dim")
            self.txt_info.insert(tk.END, str(self.image_array[:limit, :limit]))
        elif rank == 3:
            channels = ['Red', 'Green', 'Blue']
            for i in range(min(3, shape[2])):
                self.txt_info.insert(tk.END, f"\n[Kênh {channels[i]}]\n", "dim")
                self.txt_info.insert(tk.END, str(self.image_array[:limit, :limit, i]))
        
        self.txt_info.config(state='disabled')

if __name__ == "__main__":
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except: pass
    
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()
