import tkinter as tk
from tkinter import filedialog, messagebox
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class NaiveBayesPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ứng dụng Dự đoán Mô hình Naïve Bayes")
        self.root.geometry("550x450")
        
        # Biến lưu trữ dữ liệu và mô hình
        self.model = None
        self.X_test = None
        self.y_test = None
        
        # --- UI Elements ---
        
        # Tiêu đề
        tk.Label(root, text="ĐÁNH GIÁ MÔ HÌNH", font=("Arial", 16, "bold")).pack(pady=10)

        # Nút 1: Load mô hình (.joblib)
        self.btn_load_model = tk.Button(root, text="1. Tải Mô hình (.joblib)", command=self.load_model, width=35, height=2)
        self.btn_load_model.pack(pady=10)
        
        self.lbl_model_status = tk.Label(root, text="Chưa tải mô hình", fg="red")
        self.lbl_model_status.pack()
        
        # Nút 2: Load dữ liệu X_test y_test (.csv)
        self.btn_load_data = tk.Button(root, text="2. Tải Dữ liệu Test (.csv/.npy)", command=self.load_data, width=35, height=2)
        self.btn_load_data.pack(pady=10)
        
        self.lbl_data_status = tk.Label(root, text="Chưa tải dữ liệu", fg="red")
        self.lbl_data_status.pack()
        
        # Nút Dự đoán
        self.btn_predict = tk.Button(root, text="Thực hiện Dự đoán", command=self.predict_and_evaluate, width=35, height=2, state=tk.DISABLED, bg="#d9edf7", font=("Arial", 10, "bold"))
        self.btn_predict.pack(pady=20)
        
        # Khu vực hiển thị kết quả
        self.lbl_result = tk.Label(root, text="Kết quả:\nMAE: -\nMSE: -\nRMSE: -\nR2: -", justify=tk.LEFT, font=("Arial", 14, "bold"), fg="blue")
        self.lbl_result.pack(pady=10)

    def load_model(self):
        filepath = filedialog.askopenfilename(title="Chọn file mô hình", filetypes=[("Joblib Files", "*.joblib"), ("All Files", "*.*")])
        if filepath:
            try:
                self.model = joblib.load(filepath)
                self.lbl_model_status.config(text=f"Đã tải mô hình: {filepath.split('/')[-1]}", fg="green")
                self.check_ready()
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể tải mô hình:\n{str(e)}")

    def load_data(self):
        filepath = filedialog.askopenfilename(title="Chọn file dữ liệu test (Full/X&Y chung)", filetypes=[("CSV Files", "*.csv"), ("Numpy Files", "*.npy"), ("All Files", "*.*")])
        if filepath:
            try:
                if filepath.endswith('.csv'):
                    df = pd.read_csv(filepath)
                elif filepath.endswith('.npy'):
                    arr = np.load(filepath)
                    df = pd.DataFrame(arr)
                else:
                    return

                # Giả định cột cuối là y_test, các cột trước là X_test
                self.X_test = df.iloc[:, :-1]
                self.y_test = df.iloc[:, -1]
                
                self.lbl_data_status.config(text=f"Đã tải dữ liệu: {filepath.split('/')[-1]} (Dòng: {df.shape[0]})", fg="green")
                self.check_ready()
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể tải tệp dữ liệu:\n{str(e)}")

    def check_ready(self):
        if self.model is not None and self.X_test is not None and self.y_test is not None:
            self.btn_predict.config(state=tk.NORMAL)

    def predict_and_evaluate(self):
        try:
            # Thực hiện dự đoán
            y_pred = self.model.predict(self.X_test)
            
            # Tính toán các độ đo hồi quy bằng scikit-learn
            mae = mean_absolute_error(self.y_test, y_pred)
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(self.y_test, y_pred)
            
            # Hiển thị ra giao diện
            result_text = f"Kết quả Đánh giá:\n\nMAE: {mae:.4f}\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}\nR2: {r2:.4f}"
            self.lbl_result.config(text=result_text, fg="blue")
            
        except Exception as e:
            messagebox.showerror("Lỗi Dự đoán", f"Đã xảy ra lỗi khi dự đoán:\n{str(e)}\n\nLưu ý: File dữ liệu test cần có cột cuối cùng là y_test.")

if __name__ == "__main__":
    root = tk.Tk()
    app = NaiveBayesPredictorApp(root)
    root.mainloop()
