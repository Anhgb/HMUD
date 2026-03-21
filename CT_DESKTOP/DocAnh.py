import tkinter as tk
from tkinter import filedialog, ttk, messagebox, scrolledtext
from PIL import Image, ImageTk, ImageOps
import numpy as np # Import numpy for matrix handling
import matplotlib.cm as cm # Import colormap
import os

class ImageReaderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ứng dụng Xử Lý Ảnh & Phân Tích Ma Trận")
        self.root.geometry("1000x750")
        self.root.configure(bg="#f0f0f0")

        self.pil_image_original = None
        self.tk_image_original = None
        self.tk_image_gray = None

        self.setup_styles()
        self.create_layout()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TLabel", background="#f0f0f0", font=("Segoe UI", 11))
        style.configure("TButton", font=("Segoe UI", 10, "bold"))
        style.configure("Header.TLabel", font=("Segoe UI", 16, "bold"), foreground="#333")
        style.configure("Info.TLabel", background="#fff", font=("Consolas", 10))

    def create_layout(self):
        # 1. Global Header & Control
        top_frame = ttk.Frame(self.root)
        top_frame.pack(side="top", fill="x", padx=10, pady=10)

        ttk.Label(top_frame, text="CHƯƠNG TRÌNH ĐỌC & PHÂN TÍCH ẢNH", style="Header.TLabel").pack(pady=(0, 10))

        control_frame = ttk.LabelFrame(top_frame, text="Điều khiển", padding=10)
        control_frame.pack(fill="x")

        self.path_var = tk.StringVar()
        ttk.Entry(control_frame, textvariable=self.path_var, width=60).pack(side="left", padx=(0, 10), fill="x", expand=True)
        ttk.Button(control_frame, text="📂 Chọn ảnh", command=self.browse_file).pack(side="left", padx=5)
        ttk.Button(control_frame, text="📥 Tải ảnh", command=self.load_image).pack(side="left", padx=5)

        # 2. Tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # Tab 1: View
        self.tab_view = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_view, text="Hiển thị Ảnh")
        self.setup_tab_view()

        # Tab 2: Analysis (New)
        self.tab_analysis = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_analysis, text="Phân tích Ma trận/Tensor")
        self.setup_tab_analysis()

        # Tab 3: Grayscale
        self.tab_gray = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_gray, text="Chuyển đổi Xám")
        self.setup_tab_gray()

        # Tab 4: Pseudo Color (New)
        self.tab_color = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_color, text="Tô màu giả (Pseudo Color)")
        self.setup_tab_color()

    def setup_tab_view(self):
        container = ttk.Frame(self.tab_view, padding=10)
        container.pack(fill="both", expand=True)
        
        # Left: Image
        self.frame_view_original = ttk.LabelFrame(container, text="Hình ảnh", padding=5)
        self.frame_view_original.pack(side="left", fill="both", expand=True, padx=(0, 10))
        self.lbl_view_original = ttk.Label(self.frame_view_original, text="Vui lòng tải ảnh...", anchor="center")
        self.lbl_view_original.pack(fill="both", expand=True)

        # Right: Basic Info
        self.frame_info = ttk.LabelFrame(container, text="Thông tin cơ bản", padding=5)
        self.frame_info.pack(side="right", fill="y", ipadx=20)
        self.lbl_info_text = ttk.Label(self.frame_info, text="...", style="Info.TLabel", justify="left")
        self.lbl_info_text.pack(fill="both", expand=True)

    def setup_tab_analysis(self):
        container = ttk.Frame(self.tab_analysis, padding=10)
        container.pack(fill="both", expand=True)

        # Logic Button
        btn_analyze = ttk.Button(container, text="� Phân tích Chi tiết Ma trận/Tensor", command=self.analyze_data)
        btn_analyze.pack(anchor="w", pady=(0, 10))

        # Text Output
        self.txt_analysis = scrolledtext.ScrolledText(container, font=("Consolas", 10), state='normal')
        self.txt_analysis.pack(fill="both", expand=True)

    def setup_tab_gray(self):
        container = ttk.Frame(self.tab_gray, padding=10)
        container.pack(fill="both", expand=True)
        
        ttk.Button(container, text="🔄 Chuyển sang Grayscale", command=self.process_grayscale).pack(anchor="w", pady=(0, 10))
        
        self.lbl_gray_res = ttk.Label(container, text="Kết quả sẽ hiển thị ở đây", anchor="center", relief="sunken")
        self.lbl_gray_res.pack(fill="both", expand=True)

    def setup_tab_color(self):
        container = ttk.Frame(self.tab_color, padding=10)
        container.pack(fill="both", expand=True)
        
        control_subframe = ttk.Frame(container)
        control_subframe.pack(fill="x", pady=(0, 10))

        ttk.Label(control_subframe, text="Chọn bảng màu (Colormap):").pack(side="left", padx=(0, 5))
        
        self.cmap_var = tk.StringVar(value="jet")
        cmaps = ['jet', 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'hot', 'cool', 'spring', 'summer', 'autumn', 'winter']
        self.cmb_cmap = ttk.Combobox(control_subframe, textvariable=self.cmap_var, values=cmaps, state="readonly", width=15)
        self.cmb_cmap.pack(side="left", padx=5)

        ttk.Button(control_subframe, text="🎨 Chuyển sang Ảnh màu", command=self.process_pseudocolor).pack(side="left", padx=5)

        self.lbl_color_res = ttk.Label(container, text="Kết quả sẽ hiển thị ở đây", anchor="center", relief="sunken")
        self.lbl_color_res.pack(fill="both", expand=True)

    def browse_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif"), ("All files", "*.*")]
        )
        if file_path:
            self.path_var.set(file_path)
            self.load_image()

    def load_image(self):
        path = self.path_var.get()
        if not path or not os.path.exists(path):
            messagebox.showwarning("Lỗi", "Đường dẫn không hợp lệ!")
            return

        try:
            self.pil_image_original = Image.open(path)
            
            # --- Update Tab 1 ---
            img = self.pil_image_original
            info = f"File: {os.path.basename(path)}\n"
            info += f"Size: {img.width} x {img.height}\n"
            info += f"Format: {img.format}\n"
            info += f"Mode: {img.mode}\n"
            self.lbl_info_text.config(text=info)

            self.tk_image_original = self.resize_for_display(img, 600, 500)
            self.lbl_view_original.config(image=self.tk_image_original, text="")
            
            # Switch to Tab 1
            self.notebook.select(self.tab_view)
            
            # Reset other tabs
            self.txt_analysis.delete(1.0, tk.END)
            self.txt_analysis.delete(1.0, tk.END)
            self.lbl_gray_res.config(image="", text="Chưa chuyển đổi")
            self.lbl_color_res.config(image="", text="Chưa chuyển đổi")

        except Exception as e:
            messagebox.showerror("Lỗi", str(e))

    def analyze_data(self):
        if not self.pil_image_original:
            messagebox.showwarning("Cảnh báo", "Chưa có ảnh nào được tải!")
            return

        self.txt_analysis.delete(1.0, tk.END)
        img = self.pil_image_original
        
        # Helper to print to text widget
        def log(text, tag=None):
            self.txt_analysis.insert(tk.END, text + "\n", tag)

        self.txt_analysis.tag_config("header", foreground="blue", font=("Consolas", 11, "bold"))
        self.txt_analysis.tag_config("matrix", foreground="#333")

        log(f"=== PHÂN TÍCH ẢNH: {os.path.basename(self.path_var.get())} ===", "header")
        log(f"1. Kích thước (Rộng x Cao): {img.width} x {img.height}\n")

        # Convert to Numpy Array
        img_array = np.array(img)

        # Check Mode
        if img.mode == 'RGB':
            log("2. ẢNH MÀU (RGB)", "header")
            log(f"Shape Tensor: {img_array.shape} (Cao, Rộng, Kênh)\n")

            channels = ['Red (R)', 'Green (G)', 'Blue (B)']
            for i, channel_name in enumerate(channels):
                log(f"--- Kênh {channel_name} ---")
                channel_matrix = img_array[:, :, i]
                log(f"Kích thước: {channel_matrix.shape}")
                log("Ma trận (Góc 10x10 đầu tiên):")
                log(str(channel_matrix[:10, :10])) # Show snippet
                log("...\n", "matrix")

        elif img.mode == 'L':
            log("2. ẢNH ĐEN TRẮNG (GRAYSCALE)", "header")
            log(f"Shape Ma trận: {img_array.shape} (Cao, Rộng)\n")
            log("Ma trận ảnh (Góc 10x10 đầu tiên):")
            log(str(img_array[:10, :10]))
            log("...\n", "matrix")
        
        else:
            log(f"2. KHÁC ({img.mode})", "header")
            log(f"Shape: {img_array.shape}")
            log("Dữ liệu thô (Snippet):")
            log(str(img_array[:10, :10]))

        # Automatically switch to analysis tab
        self.notebook.select(self.tab_analysis)

    def process_grayscale(self):
        if not self.pil_image_original: return
        try:
            gray_img = ImageOps.grayscale(self.pil_image_original)
            self.tk_image_gray = self.resize_for_display(gray_img, 600, 500)
            self.lbl_gray_res.config(image=self.tk_image_gray, text="")
        except Exception as e:
            messagebox.showerror("Lỗi", str(e))

    def process_pseudocolor(self):
        if not self.pil_image_original: return
        try:
            # 1. Convert to Gray first (to get intensity)
            gray_img = ImageOps.grayscale(self.pil_image_original)
            gray_array = np.array(gray_img)

            # 2. Normalize to 0-1 for matplotlib
            normalized_gray = gray_array / 255.0

            # 3. Get colormap
            cmap_name = self.cmap_var.get()
            colormap = cm.get_cmap(cmap_name)

            # 4. Apply colormap (returns RGBA)
            colored_rgba = colormap(normalized_gray)

            # 5. Convert to compact uint8 (0-255)
            colored_uint8 = (colored_rgba * 255).astype(np.uint8)

            # 6. Create PIL Image (drop Alpha if not needed, but keep it is fine)
            colored_img = Image.fromarray(colored_uint8)
            # If we want RGB, we can convert
            if colored_img.mode == 'RGBA':
                colored_img = colored_img.convert('RGB')

            self.tk_image_color = self.resize_for_display(colored_img, 600, 500)
            self.lbl_color_res.config(image=self.tk_image_color, text="")
        except Exception as e:
            messagebox.showerror("Lỗi", str(e))

    def resize_for_display(self, pil_img, max_w, max_h):
        width, height = pil_img.size
        if width == 0 or height == 0: return None
        ratio = min(max_w/width, max_h/height)
        new_w = int(width * ratio)
        new_h = int(height * ratio)
        method = Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.ANTIALIAS
        resized = pil_img.resize((new_w, new_h), method)
        return ImageTk.PhotoImage(resized)

if __name__ == "__main__":
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except: pass
    root = tk.Tk()
    app = ImageReaderApp(root)
    root.mainloop()
