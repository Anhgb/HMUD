import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
import joblib
import threading
import time
import os
import cv2
from PIL import Image, ImageTk

# --- TÙY CHỈNH THEME (GIAO DIỆN NGẦU - DARK MODE) ---
BG_COLOR = "#1e1e24"          # Nền chính tối
PANEL_COLOR = "#2b2b36"       # Nền các panel
TEXT_COLOR = "#f5f5f5"        # Màu chữ trắng xám
ACCENT_COLOR = "#00d2ff"      # Màu xanh neon nhấn
BUTTON_BG = "#3a3a4a"
BUTTON_FG = "#ffffff"
BUTTON_ACTIVE = "#4e4e60"
FONT_TITLE = ("Segoe UI", 16, "bold")
FONT_NORMAL = ("Segoe UI", 10)
FONT_CONSOLE = ("Consolas", 10)

class DogCatTrainerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DOG & CAT AI TRAINER - PRO EDITION")
        self.geometry("900x750")
        self.configure(bg=BG_COLOR)
        self.resizable(False, False)

        # Biến lưu trữ đường dẫn
        self.path_x_train = tk.StringVar()
        self.path_y_train = tk.StringVar()
        self.path_x_test = tk.StringVar()
        self.path_y_test = tk.StringVar()
        self.path_img_predict = tk.StringVar() # Đường dẫn ảnh dự đoán

        # Biến model và scaler
        self.model = None
        self.scaler = None

        self.setup_ui()

    def setup_ui(self):
        # Thiết lập style Ttk
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TProgressbar", thickness=20, background=ACCENT_COLOR, troughcolor=BUTTON_BG)
        style.configure("TNotebook", background=BG_COLOR, borderwidth=0)
        style.configure("TNotebook.Tab", background=PANEL_COLOR, foreground=TEXT_COLOR, font=FONT_TITLE, padding=[20, 10])
        style.map("TNotebook.Tab", background=[("selected", ACCENT_COLOR)], foreground=[("selected", BG_COLOR)])
        
        # Tiêu đề chính
        header = tk.Label(self, text="🧠 DOG & CAT CLASSIFIER STUDIO", font=("Segoe UI", 22, "bold"), 
                          bg=BG_COLOR, fg=ACCENT_COLOR, pady=15)
        header.pack(fill=tk.X)

        # Tạo Notebook (Tab control)
        self.notebook = ttk.Notebook(self, style="TNotebook")
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)

        # TAB 1: SÂN HUẤN LUYỆN
        self.tab_train = tk.Frame(self.notebook, bg=BG_COLOR)
        self.notebook.add(self.tab_train, text="🚀 SÂN HUẤN LUYỆN")

        # TAB 2: NHẬN DẠNG ẢNH
        self.tab_predict = tk.Frame(self.notebook, bg=BG_COLOR)
        self.notebook.add(self.tab_predict, text="🖼️ NHẬN DẠNG ẢNH (VISION AI)")

        self.build_train_tab()
        self.build_predict_tab()

    def build_train_tab(self):
        # Khung chứa các chức năng (Chia 2 cột)
        main_frame = tk.Frame(self.tab_train, bg=BG_COLOR)
        main_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        left_panel = tk.Frame(main_frame, bg=PANEL_COLOR, bd=2, relief=tk.FLAT)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        right_panel = tk.Frame(main_frame, bg=PANEL_COLOR, bd=2, relief=tk.FLAT)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        # ================= KÈM GIAO DIỆN CỘT TRÁI (DATA & TRAIN) =================
        tk.Label(left_panel, text="Xác Định Dữ Liệu Huấn Luyện", font=FONT_TITLE, bg=PANEL_COLOR, fg=ACCENT_COLOR).pack(pady=10)
        
        self.create_file_selector(left_panel, "Chọn tập X_train (npy):", self.path_x_train)
        self.create_file_selector(left_panel, "Chọn tập y_train (npy):", self.path_y_train)

        self.btn_train = tk.Button(left_panel, text="🚀 BẮT ĐẦU HUẤN LUYỆN MODEL", font=("Segoe UI", 12, "bold"), 
                                   bg="#ff4757", fg=BUTTON_FG, activebackground="#ff6b81", 
                                   activeforeground=BUTTON_FG, relief=tk.FLAT, cursor="hand2",
                                   command=self.start_training_thread)
        self.btn_train.pack(pady=20, fill=tk.X, padx=30, ipady=5)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(left_panel, variable=self.progress_var, maximum=100, style="TProgressbar")
        self.progress_bar.pack(fill=tk.X, padx=30, pady=5)
        
        self.lbl_progress = tk.Label(left_panel, text="Tiến độ: 0%", font=FONT_NORMAL, bg=PANEL_COLOR, fg=TEXT_COLOR)
        self.lbl_progress.pack()

        # ================= KÈM GIAO DIỆN CỘT PHẢI (TEST & SAVE) =================
        tk.Label(right_panel, text="Kiểm Thử & Xuất Mô Hình", font=FONT_TITLE, bg=PANEL_COLOR, fg=ACCENT_COLOR).pack(pady=10)
        
        self.create_file_selector(right_panel, "Chọn tập X_test (npy):", self.path_x_test)
        self.create_file_selector(right_panel, "Chọn tập y_test (npy):", self.path_y_test)

        action_frame = tk.Frame(right_panel, bg=PANEL_COLOR)
        action_frame.pack(pady=20, fill=tk.X, padx=20)

        self.btn_test = tk.Button(action_frame, text="🎯 KIỂM THỬ MÔ HÌNH", font=("Segoe UI", 11, "bold"), 
                                  bg="#2ed573", fg=BUTTON_FG, activebackground="#7bed9f", 
                                  activeforeground=BUTTON_FG, relief=tk.FLAT, cursor="hand2", command=self.test_model)
        self.btn_test.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5, ipady=5)

        self.btn_save = tk.Button(action_frame, text="💾 LƯU MÔ HÌNH (PKL)", font=("Segoe UI", 11, "bold"), 
                                  bg="#1e90ff", fg=BUTTON_FG, activebackground="#70a1ff", 
                                  activeforeground=BUTTON_FG, relief=tk.FLAT, cursor="hand2", command=self.save_model)
        self.btn_save.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=5, ipady=5)

        # ================= MÀN HÌNH LOG BÊN DƯỚI =================
        log_frame = tk.Frame(self.tab_train, bg=BG_COLOR)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 10))
        
        tk.Label(log_frame, text="TERMINAL OUTPUT", font=("Segoe UI", 10, "bold"), bg=BG_COLOR, fg="#ffa502").pack(anchor=tk.W)

        self.log_area = scrolledtext.ScrolledText(log_frame, bg="#000000", fg="#00ff00", font=FONT_CONSOLE, height=10)
        self.log_area.pack(fill=tk.BOTH, expand=True)
        self.log_area.configure(state='disabled')
        
        self.log("🚀 KHỞI ĐỘNG HỆ THỐNG AI STUDIO V1.0...")
        self.log("Sân Huấn Luyện đã được khởi tạo.\n" + "-"*50)

    def build_predict_tab(self):
        # Khu vực Nạp mô hình & Nạp ảnh
        control_frame = tk.Frame(self.tab_predict, bg=PANEL_COLOR, bd=2, relief=tk.FLAT)
        control_frame.pack(fill=tk.X, pady=10, padx=20)
        
        tk.Label(control_frame, text="1. KHỞI TẠO MÔ HÌNH AI", font=FONT_TITLE, bg=PANEL_COLOR, fg=ACCENT_COLOR).pack(pady=(10, 5))
        
        self.lbl_model_status = tk.Label(control_frame, text="Trạng thái: Chưa tải (Hãy train ở Tab 1 hoặc Nạp file .pkl)", font=FONT_NORMAL, bg=PANEL_COLOR, fg="#ff4757")
        self.lbl_model_status.pack(pady=5)
        
        btn_load_model = tk.Button(control_frame, text="📂 Tải Mô Hình (.pkl)", font=("Segoe UI", 11, "bold"),
                                   bg=BUTTON_BG, fg=BUTTON_FG, activebackground=BUTTON_ACTIVE, relief=tk.FLAT, 
                                   cursor="hand2", command=self.load_external_model)
        btn_load_model.pack(pady=(5, 15), ipadx=10, ipady=3)

        tk.Label(control_frame, text="2. CUNG CẤP HÌNH ẢNH NHẬN DẠNG", font=FONT_TITLE, bg=PANEL_COLOR, fg=ACCENT_COLOR).pack(pady=(10, 5))
        
        self.create_file_selector(control_frame, "Đường dẫn ảnh (.jpg, .png):", self.path_img_predict, is_image=True)

        btn_predict_img = tk.Button(control_frame, text="🔍 KIỂM TRA MÔ HÌNH VÀ NHẬN DẠNG ẢNH NÀY", font=("Segoe UI", 14, "bold"),
                                 bg="#E65100", fg=BUTTON_FG, activebackground="#BF360C", relief=tk.FLAT, 
                                 cursor="hand2", command=self.predict_image_tab)
        btn_predict_img.pack(pady=(15, 20), ipadx=20, ipady=8)

        # Khu vực Hiển thị Kết quả
        result_frame = tk.Frame(self.tab_predict, bg=BG_COLOR)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=20)

        self.lbl_result = tk.Label(result_frame, text="KẾT QUẢ DỰ ĐOÁN: ---", font=("Segoe UI", 24, "bold"), bg=BG_COLOR, fg=TEXT_COLOR)
        self.lbl_result.pack(pady=(0, 10))

        self.lbl_img_preview = tk.Label(result_frame, text="[ KHUNG HIỂN THỊ HÌNH ẢNH ]", font=("Segoe UI", 12), bg="#000000", fg="#4e4e60", width=50, height=15)
        self.lbl_img_preview.pack(pady=5)

    def create_file_selector(self, parent, label_text, str_var, is_image=False):
        frame = tk.Frame(parent, bg=PANEL_COLOR)
        frame.pack(fill=tk.X, padx=20, pady=5) # Giảm pady
        
        tk.Label(frame, text=label_text, font=FONT_NORMAL, bg=PANEL_COLOR, fg=TEXT_COLOR).pack(anchor=tk.W)
        
        input_frame = tk.Frame(frame, bg=PANEL_COLOR)
        input_frame.pack(fill=tk.X, pady=2)
        
        entry = tk.Entry(input_frame, textvariable=str_var, font=FONT_NORMAL, bg=BUTTON_BG, fg=TEXT_COLOR, 
                         insertbackground=TEXT_COLOR, relief=tk.FLAT)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4, padx=(0, 5))
        
        btn = tk.Button(input_frame, text="📁 Browse", bg=BUTTON_ACTIVE, fg=TEXT_COLOR, relief=tk.FLAT, cursor="hand2",
                        command=lambda: self.browse_image(str_var) if is_image else self.browse_file(str_var))
        btn.pack(side=tk.RIGHT, ipady=1, ipadx=5)

    def browse_image(self, string_var):
        filename = filedialog.askopenfilename(title="Chọn Ảnh", filetypes=(("Image files", "*.jpg;*.jpeg;*.png"), ("All files", "*.*")))
        if filename:
            string_var.set(filename)

    def browse_file(self, string_var):
        filename = filedialog.askopenfilename(title="Chọn tập tin NPY", filetypes=(("Numpy files", "*.npy"), ("All files", "*.*")))
        if filename:
            string_var.set(filename)

    def log(self, message):
        self.log_area.configure(state='normal')
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.see(tk.END)
        self.log_area.configure(state='disabled')

    def start_training_thread(self):
        x_path = self.path_x_train.get()
        y_path = self.path_y_train.get()

        if not os.path.exists(x_path) or not os.path.exists(y_path):
            messagebox.showerror("Lỗi", "Đường dẫn file train không hợp lệ hoặc trống!")
            return

        self.btn_train.config(state=tk.DISABLED, text="⏳ ĐANG HUẤN LUYỆN...")
        self.progress_var.set(0)
        self.lbl_progress.config(text="Tiến độ: 0%")

        threading.Thread(target=self.train_process, args=(x_path, y_path), daemon=True).start()

    def train_process(self, x_path, y_path):
        try:
            self.log(f"Đang nạp tập huấn luyện từ: {x_path}")
            X_train = np.load(x_path)
            y_train = np.load(y_path)
            self.log(f"Kích thước X_train: {X_train.shape} | y_train: {y_train.shape}")

            self.log("Đang áp dụng MinMaxScaler...")
            self.scaler = MinMaxScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)

            self.log("Sử dụng kỹ thuật Warm Start (MLPClassifier) để theo dõi tiến trình...")
            classes = np.unique(y_train)
            self.model = MLPClassifier(hidden_layer_sizes=(100,), warm_start=True, max_iter=1, random_state=42, solver='adam')
            
            total_epochs = 200
            for epoch in range(total_epochs):
                self.model.partial_fit(X_train_scaled, y_train, classes=classes)
                percent = int(((epoch + 1) / total_epochs) * 100)
                self.after(0, self.update_progress, percent)
                time.sleep(0.01)
                
            self.after(0, self.log, "✅ HUẤN LUYỆN THÀNH CÔNG! Mô hình đã sẵn sàng.")
            
        except Exception as e:
            self.after(0, self.log, f"❌ LỖI TRONG QUÁ TRÌNH HUẤN LUYỆN: {str(e)}")
            self.after(0, lambda: messagebox.showerror("Lỗi", str(e)))
        finally:
            self.after(0, self.finish_training_ui)
            
    def update_progress(self, percent):
        self.progress_var.set(percent)
        self.lbl_progress.config(text=f"Tiến độ: {percent}%")
        self.update_idletasks()

    def finish_training_ui(self):
        self.btn_train.config(state=tk.NORMAL, text="🚀 BẮT ĐẦU HUẤN LUYỆN MODEL")
        self.lbl_model_status.config(text="✅ Mô hình hiện tại: Đã huấn luyện xong ở Tab 1!", fg="#2ed573")

    def test_model(self):
        if self.model is None or self.scaler is None:
            messagebox.showwarning("Cảnh báo", "Bạn chưa huấn luyện mô hình!")
            return

        x_path = self.path_x_test.get()
        y_path = self.path_y_test.get()

        if not os.path.exists(x_path) or not os.path.exists(y_path):
            messagebox.showerror("Lỗi", "Đường dẫn file test không hợp lệ hoặc trống!")
            return

        self.log(f"\nĐang nạp tập kiểm thử từ: {x_path}")
        try:
            X_test = np.load(x_path)
            y_test = np.load(y_path)
            
            self.log("Đang tiến hành đánh giá mô hình...")
            X_test_scaled = self.scaler.transform(X_test)
            
            accuracy = self.model.score(X_test_scaled, y_test)
            self.log("=" * 40)
            self.log(f"🔥 ĐỘ CHÍNH XÁC (ACCURACY): {accuracy * 100:.2f}%")
            self.log("=" * 40)
            
            sample_size = min(5, len(y_test))
            preds = self.model.predict(X_test_scaled[:sample_size])
            self.log(f"Nhãn mẫu thực tế: {y_test[:sample_size]}")
            self.log(f"Nhãn máy dự đoán: {preds}\n")

        except Exception as e:
            self.log(f"❌ LỖI KIỂM THỬ: {str(e)}")

    def load_external_model(self):
        filepath = filedialog.askopenfilename(title="Chọn file mô hình", filetypes=[("Pickle File", "*.pkl"), ("All files", "*.*")])
        if not filepath:
            return
        try:
            data = joblib.load(filepath)
            if isinstance(data, dict) and 'model' in data:
                self.model = data.get('model')
                self.scaler = data.get('scaler')
            else:
                self.model = data
                self.scaler = None
                
            filename = os.path.basename(filepath)
            self.lbl_model_status.config(text=f"✅ Mô hình nạp từ: {filename}", fg="#2ed573")
            messagebox.showinfo("Thành công", f"Đã nạp thành công mô hình: {filename}")
        except Exception as e:
            messagebox.showerror("Lỗi Nạp", f"Không thể nạp mô hình.\nChi tiết: {e}")

    def predict_image_tab(self):
        if self.model is None:
            messagebox.showwarning("Cảnh báo", "Hệ thống chưa có mô hình nhận dạng! Vui lòng Train mô hình ở Tab 1 hoặc Nạp file .pkl")
            return
            
        filepath = self.path_img_predict.get()
        if not filepath or not os.path.exists(filepath):
            messagebox.showerror("Lỗi", "Vui lòng chọn hoặc nhập đường dẫn ảnh hợp lệ trước khi nhận dạng!")
            return
            
        try:
            img = cv2.imread(filepath)
            if img is None:
                messagebox.showerror("Lỗi", "Không thể đọc ảnh. File có thể bị hỏng/không đúng định dạng.")
                return
                
            # Tính kích thước zxz dựa vào n_features của mô hình (thay vì scaler)
            z_squared = getattr(self.model, 'n_features_in_', None)
            if z_squared is None:
                messagebox.showerror("Lỗi", "Mô hình này không tương thích (không rõ số pixels đầu vào).")
                return
                
            z = int(np.sqrt(z_squared))
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (z, z))
            vector = resized.flatten().reshape(1, -1)
            
            if self.scaler is not None and hasattr(self.scaler, 'transform'):
                vector_scaled = self.scaler.transform(vector)
            else:
                vector_scaled = vector / 255.0
            
            pred = self.model.predict(vector_scaled)[0]
            
            res_str = str(pred).strip().lower()
            if res_str in ["1", "1.0", "dog", "chó"]:
                label_text = "DOG - Chó 🐶"
                color = "#2ed573"
            elif res_str in ["0", "0.0", "cat", "mèo"]:
                label_text = "CAT - Mèo 🐱"
                color = "#ff4757"
            else:
                label_text = f"Nhãn: {pred}"
                color = "#f39c12"
            
            # Hiển thị ảnh kèm popup
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)
            
            # Thay đổi kích thước ảnh hiển thị cho vừa vặn (thu nhỏ lại để đỡ tràn khung)
            pil_img.thumbnail((250, 250), Image.Resampling.LANCZOS)
            tk_img = ImageTk.PhotoImage(pil_img)
            
            self.lbl_img_preview.config(image=tk_img, text="", width=pil_img.width, height=pil_img.height)
            self.lbl_img_preview.image = tk_img # Keep a reference!
            
            self.lbl_result.config(text=f"KẾT QUẢ DỰ ĐOÁN: {label_text}", fg=color)
            
        except Exception as e:
            messagebox.showerror("Lỗi xử lý ảnh", str(e))

    def save_model(self):
        if self.model is None:
            messagebox.showwarning("Cảnh báo", "Không có cấu trúc mô hình để lưu!")
            return
            
        file_path = filedialog.asksaveasfilename(defaultextension=".pkl", 
                                                 filetypes=(("Pickle Model", "*.pkl"), ("Tất cả files", "*.*")),
                                                 initialfile="dog_cat_model.pkl")
        if file_path:
            try:
                joblib.dump({'model': self.model, 'scaler': self.scaler}, file_path)
                self.log(f"💾 Đã lưu cấu trúc mô hình & bộ chuẩn hóa (Scaler) tại:\n{file_path}\n")
                messagebox.showinfo("Lưu Thành Công", "Mô hình đã được lưu an toàn!")
            except Exception as e:
                self.log(f"❌ LỖI LƯU MÔ HÌNH: {str(e)}")
                messagebox.showerror("Lỗi Lưu", f"Không thể lưu: {e}")

if __name__ == "__main__":
    app = DogCatTrainerApp()
    app.mainloop()
