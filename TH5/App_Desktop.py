import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

class ModernPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Model Evaluator - Pro Editon")
        self.root.geometry("600x550")
        self.root.configure(bg="#f4f6f9") # Nền xám nhạt hiện đại
        
        # Thay đổi icon nếu có (bỏ qua nếu không có file .ico)
        # self.root.iconbitmap("icon.ico")
        
        self.model = None
        self.X_test = None
        self.y_test = None

        self.setup_styles()
        self.create_widgets()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam') # Theme clam hỗ trợ tùy chỉnh màu sắc tốt hơn trên Windows

        # Custom font
        self.font_title = ("Segoe UI", 16, "bold")
        self.font_header = ("Segoe UI", 12, "bold")
        self.font_normal = ("Segoe UI", 10)
        self.font_result = ("Consolas", 11)

        # Style cho Label
        style.configure("TLabel", background="#f4f6f9", font=self.font_normal, foreground="#333")
        style.configure("Header.TLabel", font=self.font_header, foreground="#2c3e50")
        style.configure("Status.TLabel", font=("Segoe UI", 9, "italic"), foreground="#e74c3c")
        style.configure("Success.TLabel", font=("Segoe UI", 9, "italic"), foreground="#27ae60")

        # Style cho Button
        style.configure("Primary.TButton", font=self.font_normal, padding=10, background="#3498db", foreground="white")
        style.map("Primary.TButton", background=[('active', '#2980b9')])

        style.configure("Action.TButton", font=("Segoe UI", 11, "bold"), padding=12, background="#2ecc71", foreground="white")
        style.map("Action.TButton", background=[('active', '#27ae60')])

        # Style cho Frame "Card"
        style.configure("Card.TFrame", background="#ffffff")

    def create_widgets(self):
        # --- Header Banner ---
        header_frame = tk.Frame(self.root, bg="#2c3e50", height=60)
        header_frame.pack(fill="x", side="top")
        
        lbl_title = tk.Label(header_frame, text="MODEL EVALUATION DASHBOARD", 
                             font=self.font_title, bg="#2c3e50", fg="white")
        lbl_title.pack(pady=15)

        # --- Main Layout Container ---
        container = tk.Frame(self.root, bg="#f4f6f9", padx=20, pady=20)
        container.pack(fill="both", expand=True)

        # 1. CARD: TẢI MODEL
        card1 = tk.Frame(container, bg="white", highlightbackground="#e0e0e0", highlightthickness=1, padx=15, pady=15)
        card1.pack(fill="x", pady=(0, 15))

        lbl_step1 = tk.Label(card1, text="Bước 1: Nạp Mô Hình (Joblib / Pkl)", font=self.font_header, bg="white", fg="#34495e")
        lbl_step1.pack(anchor="w", pady=(0, 10))

        btn_load_model = ttk.Button(card1, text="Tải File Mô Hình", command=self.load_model, style="Primary.TButton", cursor="hand2")
        btn_load_model.pack(side="left", padx=(0, 15))

        self.lbl_model_status = ttk.Label(card1, text="Đang đợi mô hình...", style="Status.TLabel", background="white")
        self.lbl_model_status.pack(side="left")

        # 2. CARD: TẢI DỮ LIỆU
        card2 = tk.Frame(container, bg="white", highlightbackground="#e0e0e0", highlightthickness=1, padx=15, pady=15)
        card2.pack(fill="x", pady=(0, 20))

        lbl_step2 = tk.Label(card2, text="Bước 2: Nạp Dữ Liệu Test (CSV / Excel)", font=self.font_header, bg="white", fg="#34495e")
        lbl_step2.pack(anchor="w", pady=(0, 10))

        btn_load_data = ttk.Button(card2, text="Tải Dữ Liệu Test", command=self.load_data, style="Primary.TButton", cursor="hand2")
        btn_load_data.pack(side="left", padx=(0, 15))

        self.lbl_data_status = ttk.Label(card2, text="Đang đợi dữ liệu CSV / Excel...", style="Status.TLabel", background="white")
        self.lbl_data_status.pack(side="left")

        # 3. ACTION BUTTON
        btn_predict = ttk.Button(container, text="🔥 THỰC HIỆN DỰ ĐOÁN & ĐÁNH GIÁ 🔥", command=self.predict_and_evaluate, style="Action.TButton", cursor="hand2")
        btn_predict.pack(fill="x", pady=(0, 20))

        # 4. CARD: KẾT QUẢ ĐÁNH GIÁ (Terminal-like design)
        card3 = tk.Frame(container, bg="#2d3436", highlightbackground="#636e72", highlightthickness=1, padx=15, pady=15)
        card3.pack(fill="both", expand=True)

        lbl_step3 = tk.Label(card3, text="Kết Quả Đánh Giá Hiệu Suất", font=("Segoe UI", 10, "bold"), bg="#2d3436", fg="#dfe6e9")
        lbl_step3.pack(anchor="w", pady=(0, 10))

        self.lbl_results = tk.Label(card3, text="> Sẵn sàng. Hãy tải file để bắt đầu...", font=self.font_result, bg="#2d3436", fg="#00cec9", justify="left")
        self.lbl_results.pack(anchor="w")

    def load_model(self):
        filepath = filedialog.askopenfilename(
            title="Chọn file mô hình",
            filetypes=(("Model files", "*.joblib *.pkl"), ("All files", "*.*"))
        )
        if filepath:
            try:
                # joblib có thể load được cả file .pkl và .joblib
                self.model = joblib.load(filepath)
                # Thay đổi style label sang màu xanh nhẹ
                self.lbl_model_status.config(text=f"✔ Đã tải: {filepath.split('/')[-1]}", style="Success.TLabel")
            except Exception as e:
                # Nếu joblib không load được .pkl, thử với thư viện pickle tiêu chuẩn
                try:
                    import pickle
                    with open(filepath, 'rb') as f:
                        self.model = pickle.load(f)
                    self.lbl_model_status.config(text=f"✔ Đã tải: {filepath.split('/')[-1]}", style="Success.TLabel")
                except Exception as ex:
                    messagebox.showerror("Lỗi Nạp Mô Hình", f"Chi tiết lỗi:\n{str(e)}\n{str(ex)}")

    def load_data(self):
        filepath = filedialog.askopenfilename(
            title="Chọn file dữ liệu",
            filetypes=(("Data files", "*.csv *.xlsx *.xls"), ("All files", "*.*"))
        )
        if filepath:
            try:
                # Đọc dữ liệu, dựa theo đuôi file
                if filepath.endswith('.csv'):
                    df = pd.read_csv(filepath)
                else:
                    df = pd.read_excel(filepath)

                self.X_test = df.iloc[:, :-1]
                self.y_test = df.iloc[:, -1]
                
                self.lbl_data_status.config(text=f"✔ Đã tải dataset: {filepath.split('/')[-1]} ({len(df)} hàng)", style="Success.TLabel")
            except Exception as e:
                messagebox.showerror("Lỗi Nạp Dữ Liệu", f"Kiểm tra lại định dạng file (CSV/Excel)!\nChi tiết:\n{str(e)}\n\n(Lưu ý: Nếu đọc file Excel bị lỗi missing openpyxl, bạn cần mở Terminal chạy: pip install openpyxl)")

    def predict_and_evaluate(self):
        if self.model is None:
            messagebox.showwarning("Thiếu Dữ Liệu", "Chưa tải File Mô Hình ở Bước 1!")
            return
        if self.X_test is None or self.y_test is None:
            messagebox.showwarning("Thiếu Dữ Liệu", "Chưa tải File Dữ liệu CSV ở Bước 2!")
            return

        try:
            # 1. Thực hiện dự đoán
            y_pred = self.model.predict(self.X_test)
            
            # 2. Tính toán các độ đo (dùng thư viện sklearn.metrics)
            mae = mean_absolute_error(self.y_test, y_pred)
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(self.y_test, y_pred)

            # 3. Hiển thị kết quả dạng bảng Console đẹp
            result_str = (
                f"> Hệ số MAE  : {mae:,.4f}\n"
                f"> Hệ số MSE  : {mse:,.4f}\n"
                f"> Hệ số RMSE : {rmse:,.4f}\n"
                f"> Hệ số R²   : {r2:,.4f}"
            )
            self.lbl_results.config(text=result_str, fg="#55efc4") # Màu xanh lá neon hiện đại
            
        except Exception as e:
            messagebox.showerror("Lỗi Khớp Dữ Liệu", f"Số lượng thuộc tính (cột) trong CSV không khớp với mô hình đã huấn luyện!\n\nChi tiết:\n{str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ModernPredictionApp(root)
    root.mainloop()

