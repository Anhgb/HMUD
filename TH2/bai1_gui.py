
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
import numpy as np
import os

# --- Color Palette & Font Settings ---
COLORS = {
    "bg_main": "#ECEFF1",       # Light Grey Blue Background
    "bg_card": "#FFFFFF",       # White Card Background
    "primary": "#1976D2",       # Darker Blue for Actions
    "primary_light": "#2196F3", # Lighter Blue for Hover
    "secondary": "#FF6F00",     # Deep Orange for Highlights
    "text_header": "#263238",   # Dark Blue Grey for Headers
    "text_body": "#455A64",     # Grey for Body Text
    "border_light": "#CFD8DC",  # Light Border
    "success": "#388E3C",       # Green
    "error": "#D32F2F"          # Red
}

FONTS = {
    "title": ("Helvetica", 18, "bold"),
    "subtitle": ("Helvetica", 12, "bold"),
    "label": ("Helvetica", 10),
    "entry": ("Helvetica", 10),
    "button": ("Helvetica", 10, "bold"),
    "code": ("Consolas", 10)
}

class VectorGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NumPy Vector Generator Pro")
        self.root.geometry("750x650")
        self.root.configure(bg=COLORS["bg_main"])

        self.setup_styles()
        self.create_layout()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')  # Base theme

        # --- Frames ---
        style.configure("Card.TFrame", background=COLORS["bg_card"], borderwidth=1, relief="solid", bordercolor=COLORS["border_light"])
        style.configure("Main.TFrame", background=COLORS["bg_main"])

        # --- Labels ---
        style.configure("Title.TLabel", background=COLORS["bg_main"], foreground=COLORS["text_header"], font=FONTS["title"])
        style.configure("Subtitle.TLabel", background=COLORS["bg_card"], foreground=COLORS["text_header"], font=FONTS["subtitle"])
        style.configure("Body.TLabel", background=COLORS["bg_card"], foreground=COLORS["text_body"], font=FONTS["label"])
        style.configure("Header.TLabel", background=COLORS["primary"], foreground="#FFFFFF", font=FONTS["subtitle"])

        # --- Entries ---
        style.configure("Modern.TEntry", fieldbackground=COLORS["bg_main"], foreground=COLORS["text_header"], bordercolor=COLORS["border_light"], padding=5)

        # --- Buttons ---
        # Generate Button
        style.configure("Primary.TButton", 
                        background=COLORS["primary"], 
                        foreground="white", 
                        font=FONTS["button"], 
                        borderwidth=0, 
                        focuscolor="none")
        style.map("Primary.TButton", 
                  background=[('active', COLORS["primary_light"]), ('pressed', COLORS["primary"])])

        # Clear Button
        style.configure("Secondary.TButton", 
                        background="#B0BEC5", 
                        foreground=COLORS["text_header"], 
                        font=FONTS["button"], 
                        borderwidth=0, 
                        focuscolor="none")
        style.map("Secondary.TButton", 
                  background=[('active', "#CFD8DC")])

    def create_layout(self):
        # 1. Header Banner
        header_frame = tk.Frame(self.root, bg=COLORS["bg_main"])
        header_frame.pack(fill="x", padx=20, pady=(20, 10))
        
        lbl_icon = tk.Label(header_frame, text="🔢", font=("Arial", 30), bg=COLORS["bg_main"], fg=COLORS["primary"])
        lbl_icon.pack(side="left", padx=(0, 10))
        
        lbl_title = ttk.Label(header_frame, text="NumPy Vector Automation", style="Title.TLabel")
        lbl_title.pack(side="left", fill="y")

        # 2. Main Content Container
        content_container = ttk.Frame(self.root, style="Main.TFrame")
        content_container.pack(fill="both", expand=True, padx=20, pady=10)

        # --- Card: Configuration ---
        config_card = ttk.Frame(content_container, style="Card.TFrame", padding=20)
        config_card.pack(fill="x", pady=(0, 20))

        # Title within card
        ttk.Label(config_card, text="CẤU HÌNH VECTOR TÙY CHỌN", style="Subtitle.TLabel").pack(anchor="w", pady=(0, 15))

        # Flex-grid layout for inputs
        input_container = tk.Frame(config_card, bg=COLORS["bg_card"])
        input_container.pack(fill="x")

        # Input Group function
        def create_input_group(parent, label_text, default_val, col):
            frame = tk.Frame(parent, bg=COLORS["bg_card"])
            frame.grid(row=0, column=col, padx=10, sticky="ew")
            parent.grid_columnconfigure(col, weight=1)
            
            ttk.Label(frame, text=label_text, style="Body.TLabel").pack(anchor="w", pady=(0, 5))
            entry = ttk.Entry(frame, style="Modern.TEntry", width=10)
            entry.insert(0, str(default_val))
            entry.pack(fill="x", ipady=3)
            return entry

        self.entry_n = create_input_group(input_container, "Số chiều (n)", 15, 0)
        self.entry_min = create_input_group(input_container, "Giá trị Min", -5.0, 1)
        self.entry_max = create_input_group(input_container, "Giá trị Max", 5.0, 2)

        # Action Buttons
        btn_container = tk.Frame(config_card, bg=COLORS["bg_card"])
        btn_container.pack(fill="x", pady=(20, 0))

        btn_generate = ttk.Button(btn_container, text="KHỞI TẠO & LƯU FILE", style="Primary.TButton", cursor="hand2", command=self.process_vectors)
        btn_generate.pack(side="right", padx=(10, 0), ipadx=10, ipady=5)

        btn_clear = ttk.Button(btn_container, text="Làm mới", style="Secondary.TButton", cursor="hand2", command=self.clear_results)
        btn_clear.pack(side="right", ipadx=10, ipady=5)

        # --- Card: Output ---
        output_card = ttk.Frame(content_container, style="Card.TFrame", padding=2) # Thin padding for border effect
        output_card.pack(fill="both", expand=True)

        # Fake Header for Output
        output_header = tk.Label(output_card, text="  BẢNG KẾT QUẢ CHI TIẾT", bg=COLORS["text_header"], fg="white", font=FONTS["subtitle"], anchor="w", pady=8)
        output_header.pack(fill="x")

        # Text Area
        self.txt_result = scrolledtext.ScrolledText(output_card, 
                                                    font=FONTS["code"], 
                                                    bg="#FAFAFA", 
                                                    fg=COLORS["text_body"],
                                                    relief="flat",
                                                    padx=10, pady=10)
        self.txt_result.pack(fill="both", expand=True)

        # Configure Tags
        self.txt_result.tag_config("h1", foreground=COLORS["primary"], font=("Helvetica", 11, "bold"))
        self.txt_result.tag_config("meta", foreground="#90A4AE", font=("Consolas", 9, "italic"))
        self.txt_result.tag_config("highlight", background="#E3F2FD", foreground="#0D47A1")

        # Status Bar
        self.status_var = tk.StringVar(value="Sẵn sàng")
        status_bar = tk.Label(self.root, textvariable=self.status_var, bg="#CFD8DC", fg=COLORS["text_body"], font=("Helvetica", 9), anchor="w", padx=10, pady=2)
        status_bar.pack(side="bottom", fill="x")

    def clear_results(self):
        self.txt_result.delete(1.0, tk.END)
        self.status_var.set("Đã làm mới dữ liệu")

    def process_vectors(self):
        self.status_var.set("Đang xử lý...")
        self.root.update_idletasks()
        try:
            # Validate Inputs
            try:
                n = int(self.entry_n.get())
                min_val = float(self.entry_min.get())
                max_val = float(self.entry_max.get())
            except ValueError:
                self.status_var.set("Lỗi: Dữ liệu nhập không hợp lệ")
                messagebox.showerror("Dữ Liệu Không Hợp Lệ", "Vui lòng kiểm tra lại các trường nhập liệu.\n- n: số nguyên\n- min/max: số thực")
                return

            if n <= 0:
                self.status_var.set("Lỗi: Số chiều n phải dương")
                messagebox.showwarning("Tham Số Sai", "Số chiều vector (n) phải lớn hơn 0.")
                return

            if min_val > max_val:
                min_val, max_val = max_val, min_val
                self.entry_min.delete(0, tk.END); self.entry_min.insert(0, str(min_val))
                self.entry_max.delete(0, tk.END); self.entry_max.insert(0, str(max_val))

            # Generate Data
            vector1 = np.random.randint(1, 11, 10)
            vector2 = np.random.uniform(-1, 1, 20)
            vector3 = np.random.uniform(min_val, max_val, n)

            # Display
            self.txt_result.delete(1.0, tk.END)
            
            def display_block(name, desc, data):
                self.txt_result.insert(tk.END, f"📌 {name}\n", "h1")
                self.txt_result.insert(tk.END, f"   ➤ {desc}\n", "meta")
                self.txt_result.insert(tk.END, f"{data}\n\n", "highlight")

            display_block("Vector 1", "Ngẫu nhiên số nguyên [1, 10]", vector1)
            display_block("Vector 2", "Ngẫu nhiên số thực [-1, 1]", vector2)
            display_block("Vector 3", f"Tùy chọn: n={n}, khoảng=[{min_val}, {max_val}]", vector3)

            # Save File
            self.save_to_file(vector1, vector2, vector3, n, min_val, max_val)
            self.status_var.set("Hoàn tất: Đã lưu file vectors.txt")

        except Exception as e:
            self.status_var.set("Lỗi hệ thống")
            messagebox.showerror("Lỗi", str(e))

    def save_to_file(self, v1, v2, v3, n, min_v, max_v):
        try:
            filename = "vectors.txt"
            filepath = os.path.join(os.path.dirname(__file__), filename)
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("=== VECTOR AUTOMATION REPORT ===\n\n")
                
                f.write("> Vector 1 (Integers 1-10):\n")
                np.savetxt(f, v1.reshape(1, -1), fmt='%d', delimiter=', ')
                
                f.write("\n> Vector 2 (Floats -1 to 1):\n")
                np.savetxt(f, v2.reshape(1, -1), fmt='%.4f', delimiter=', ')
                
                f.write(f"\n> Vector 3 (Custom n={n}, range=[{min_v}, {max_v}]):\n")
                np.savetxt(f, v3.reshape(1, -1), fmt='%.4f', delimiter=', ')
                
        except Exception as e:
            raise e

if __name__ == "__main__":
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1) # Cải thiện độ sắc nét trên màn hình High DPI
    except:
        pass
        
    root = tk.Tk()
    app = VectorGeneratorApp(root)
    root.mainloop()
