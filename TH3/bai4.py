import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from PIL import Image, ImageTk
import numpy as np
import sys
import os
import math

# Redirect stdout to handle unicode if needed
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

class ImageGuiApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bài 4: Xử Lý Ảnh OOP (Modern UI)")
        self.root.geometry("1100x700")
        self.root.configure(bg="#f3f4f6") # Light gray background

        # Initialize Data Attributes
        self.original_image = None
        self.current_image = None
        self.image_array = None
        
        # Radar Mode State
        self.is_radar_active = False
        self.radar_base_image = None
        self.update_counter = 0 # Throttle for info updates

        # Setup Styles & UI
        self.setup_styles()
        self.setup_ui()
        self.setup_menu()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam') # Good base for custom styling

        # Colors
        bg_color = "#f3f4f6"
        accent_color = "#2563eb" # Blue
        text_color = "#1f2937"
        header_bg = "#ffffff"
        
        # General Label Style
        style.configure("TLabel", background=header_bg, foreground=text_color, font=("Segoe UI", 10))
        
        # Frame Style
        style.configure("Card.TFrame", background=header_bg, relief="flat")
        
        # Button Style
        style.configure("TButton", 
                        font=("Segoe UI", 10, "bold"), 
                        background=accent_color, 
                        foreground="white", 
                        borderwidth=0, 
                        focuscolor=accent_color)
        style.map("TButton", background=[("active", "#1d4ed8")]) # Darker blue on hover

        # Heading Style
        style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"), foreground="#111827", background=header_bg)
        
        # Subheading Style
        style.configure("SubHeader.TLabel", font=("Segoe UI", 11, "bold"), foreground="#374151", background=header_bg)

    def setup_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # 1. File Menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Hệ thống", menu=file_menu)
        file_menu.add_command(label="Mở ảnh...", command=self.open_image)
        file_menu.add_separator()
        file_menu.add_command(label="Thoát", command=self.root.quit)

        # 2. Process Menu
        process_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Xử lý", menu=process_menu)
        process_menu.add_command(label="Chuyển sang ảnh xám", command=self.process_grayscale)
        process_menu.add_command(label="Chuyển vị ảnh", command=self.process_transpose)
        process_menu.add_command(label="Reset về ảnh gốc", command=self.reset_image)

        # 3. View Menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Hiển thị", menu=view_menu)
        view_menu.add_command(label="Cập nhật thông tin", command=self.update_info_display)

    def setup_ui(self):
        # Main Layout: uses grid or pack with padding
        main_container = ttk.Frame(self.root, style="Card.TFrame")
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Title / Header
        header_frame = ttk.Frame(main_container, style="Card.TFrame")
        header_frame.pack(fill=tk.X, padx=20, pady=(20, 10))
        ttk.Label(header_frame, text="CHƯƠNG TRÌNH XỬ LÝ ẢNH", style="Header.TLabel").pack(side=tk.LEFT)
        
        # Toolbar
        toolbar_frame = ttk.Frame(main_container, style="Card.TFrame")
        toolbar_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        ttk.Button(toolbar_frame, text="📂 Mở Ảnh", command=self.open_image).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(toolbar_frame, text="🔄 Reset", command=self.reset_image).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(toolbar_frame, text="🌑 Grayscale", command=self.process_grayscale).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(toolbar_frame, text="🎨 Màu Gốc", command=self.process_restore_color).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(toolbar_frame, text="🌀 Radar Rotate", command=self.toggle_radar_mode).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(toolbar_frame, text="📐 Chuyển Vị", command=self.process_transpose).pack(side=tk.LEFT)

        # Split Layout (PanedWindow)
        self.paned_window = tk.PanedWindow(main_container, orient=tk.HORIZONTAL, bg="#e5e7eb", sashwidth=4, sashrelief="flat")
        self.paned_window.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

        # LEFT PANE: Image
        self.frame_left = tk.Frame(self.paned_window, bg="white")
        self.paned_window.add(self.frame_left)
        
        # Image Container (Center Alignment)
        self.lbl_image_container = tk.Label(self.frame_left, text="Vui lòng mở ảnh để bắt đầu...", bg="white", fg="#9ca3af", font=("Segoe UI", 12))
        self.lbl_image_container.pack(expand=True, fill=tk.BOTH, padx=2, pady=2)
        
        # Bind mouse movement for radar mode
        self.lbl_image_container.bind('<Motion>', self.on_mouse_move)

        # RIGHT PANE: Info
        self.frame_right = tk.Frame(self.paned_window, bg="white")
        self.paned_window.add(self.frame_right)

        right_content = ttk.Frame(self.frame_right, style="Card.TFrame", padding=15)
        right_content.pack(fill=tk.BOTH, expand=True)

        ttk.Label(right_content, text="THÔNG TIN CHI TIẾT", style="SubHeader.TLabel").pack(anchor="w", pady=(0, 10))
        
        self.txt_info = scrolledtext.ScrolledText(right_content, wrap=tk.NONE, font=("Consolas", 10), state='disabled', relief="flat", bg="#f9fafb", padx=10, pady=10)
        self.txt_info.pack(fill=tk.BOTH, expand=True)

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

    def reset_image(self):
        if self.original_image:
            self.current_image = self.original_image.copy()
            self.image_array = np.array(self.original_image)
            self.display_image(self.current_image)
            self.update_info_display()

    def display_image(self, pil_img):
        if not pil_img: return

        # Intelligent Resize
        container_w = self.frame_left.winfo_width()
        container_h = self.frame_left.winfo_height()
        
        # Fallback if window not fully drawn
        if container_w < 10: container_w = 600
        if container_h < 10: container_h = 500
        
        # Fit image within container with padding
        target_size = (container_w - 20, container_h - 20)
        
        img_copy = pil_img.copy()
        img_copy.thumbnail(target_size, Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.ANTIALIAS)
        
        self.tk_image = ImageTk.PhotoImage(img_copy)
        self.lbl_image_container.config(image=self.tk_image, text="")
        self.lbl_image_container.image = self.tk_image

    def toggle_radar_mode(self):
        """Toggle the radar rotation mode on/off."""
        if not self.current_image:
            messagebox.showwarning("Cảnh báo", "Vui lòng mở ảnh trước!")
            return
            
        self.is_radar_active = not self.is_radar_active
        
        if self.is_radar_active:
            # Prepare base image for rotation (resized to fit current view to avoid lag)
            container_w = self.frame_left.winfo_width()
            container_h = self.frame_left.winfo_height()
            target_size = (container_w - 20, container_h - 20)
            
            self.radar_base_image = self.current_image.copy()
            self.radar_base_image.thumbnail(target_size, Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.ANTIALIAS)
            messagebox.showinfo("Radar Mode", "Chế độ Radar đã BẬT. Di chuyển chuột để xoay ảnh.")
        else:
            # Restore normal view
            self.display_image(self.current_image)
            messagebox.showinfo("Radar Mode", "Chế độ Radar đã TẮT.")

    def on_mouse_move(self, event):
        """Rotate image based on mouse position relative to center."""
        if not self.is_radar_active or not self.radar_base_image:
            return
            
        # Get center of the label/container
        w = self.lbl_image_container.winfo_width()
        h = self.lbl_image_container.winfo_height()
        center_x = w // 2
        center_y = h // 2
        
        # Calculate angle
        dx = event.x - center_x
        dy = event.y - center_y
        
        # atan2 returns radians, convert to degrees
        # -degrees because PIL rotate is counter-clockwise and we want intuitive 'look at' behavior
        # Adding 90 or adjusting might be needed depending on initial orientation, 
        # but pure rotation is usually requested. Let's try direct mapping first.
        angle = math.degrees(math.atan2(dy, dx))
        
        # Rotate image (expand=True might resize container, let's keep it False to rotate in place)
        rotated_img = self.radar_base_image.rotate(-angle)
        
        self.tk_image = ImageTk.PhotoImage(rotated_img)
        self.lbl_image_container.config(image=self.tk_image)
        self.lbl_image_container.image = self.tk_image
        
        # Real-time Parameter Update (Throttled)
        self.update_counter += 1
        if self.update_counter % 5 == 0: # Update every 5th movement event approx
             self.image_array = np.array(rotated_img)
             self.update_info_display()

    def process_restore_color(self):
        """Khôi phục lại ảnh màu gốc từ ảnh xám hoặc ảnh đã biến đổi."""
        if self.original_image:
            # Revert to original image
            self.current_image = self.original_image.copy()
            self.image_array = np.array(self.current_image)
            self.display_image(self.current_image)
            self.update_info_display()
        else:
            messagebox.showwarning("Cảnh báo", "Vui lòng mở ảnh trước!")

    def process_grayscale(self):
        if self.current_image:
            self.current_image = self.current_image.convert('L')
            self.image_array = np.array(self.current_image)
            self.display_image(self.current_image)
            self.update_info_display()
        else:
            messagebox.showwarning("Cảnh báo", "Vui lòng mở ảnh trước!")

    def process_transpose(self):
        # Stop any active rotation modes
        self.is_radar_active = False
        # self.is_auto_rotate = False # This variable is not defined in the original code.

        if self.current_image:
            # Ensure we are working with the full resolution image, not the radar thumbnail
            self.image_array = np.array(self.current_image)
            
            rank = self.image_array.ndim
            if rank == 3:
                transposed = np.transpose(self.image_array, (1, 0, 2))
            else:
                transposed = self.image_array.T
            
            self.image_array = transposed
            try:
                self.current_image = Image.fromarray(self.image_array)
                self.display_image(self.current_image)
                self.update_info_display()
            except Exception as e:
                messagebox.showwarning("Lỗi hiển thị", f"Không thể hiển thị ma trận này dưới dạng ảnh: {e}")
        else:
            messagebox.showwarning("Cảnh báo", "Vui lòng mở ảnh trước!")

    def update_info_display(self):
        if self.image_array is None: return

        self.txt_info.config(state='normal')
        self.txt_info.delete(1.0, tk.END)

        shape = self.image_array.shape
        rank = self.image_array.ndim
        dtype = self.image_array.dtype

        # Styles for text widget tags
        self.txt_info.tag_config("bold", font=("Consolas", 10, "bold"), foreground="#2563eb")
        self.txt_info.tag_config("dim", foreground="#6b7280")

        self.txt_info.insert(tk.END, "THÔNG SỐ TENSOR\n", "bold")
        self.txt_info.insert(tk.END, f"• Hạng (Rank): {rank}\n")
        self.txt_info.insert(tk.END, f"• Kích thước:  {shape}\n")
        self.txt_info.insert(tk.END, f"• Kiểu dữ liệu:{dtype}\n\n")
        
        self.txt_info.insert(tk.END, "DỮ LIỆU MA TRẬN (Góc 20x20)\n", "bold")

        limit = 20
        
        if rank == 2:
            self.txt_info.insert(tk.END, "[Ảnh Đen Trắng]\n", "dim")
            self.txt_info.insert(tk.END, str(self.image_array[:limit, :limit]))
        elif rank == 3:
            channels = ['Red', 'Green', 'Blue', 'Alpha']
            num_channels = shape[2]
            for i in range(num_channels):
                c_name = channels[i] if i < len(channels) else f"Ch.{i}"
                self.txt_info.insert(tk.END, f"\n[Kênh: {c_name}]\n", "dim")
                self.txt_info.insert(tk.END, str(self.image_array[:limit, :limit, i]))
        
        self.txt_info.config(state='disabled')


if __name__ == "__main__":
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1) # Enable DPI awareness on Windows
    except:
        pass
        
    root = tk.Tk()
    app = ImageGuiApp(root)
    root.mainloop()
