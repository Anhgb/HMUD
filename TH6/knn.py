import customtkinter as ctk
from tkinter import filedialog, messagebox
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)

ctk.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class PremiumPredictorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Cấu hình cửa sổ chính
        self.title("AI Premium Model Predictor")
        self.geometry("900x700")
        self.resizable(True, True)

        # Tiêu đề chính
        self.lbl_title = ctk.CTkLabel(self, text="HỆ THỐNG DỰ ĐOÁN MÔ HÌNH AI", font=ctk.CTkFont(size=24, weight="bold"))
        self.lbl_title.pack(pady=(20, 10))

        # Khởi tạo Tabview
        self.tabview = ctk.CTkTabview(self, width=800, height=550)
        self.tabview.pack(padx=20, pady=10, fill="both", expand=True)

        # Thêm 3 Tab
        self.tab_knn = self.tabview.add("Mô hình KNN")
        self.tab_reg = self.tabview.add("Mô hình Hồi quy")
        self.tab_nb = self.tabview.add("Mô hình Naïve Bayes")

        # --- BIẾN LƯU TRỮ TRẠNG THÁI ---
        # Data cho Tab KNN
        self.model_knn = None
        self.X_test_knn = None
        self.y_test_knn = None

        # Data cho Tab Hồi quy
        self.model_reg = None
        self.X_test_reg = None
        self.y_test_reg = None

        # Data cho Tab Naïve Bayes
        self.model_nb = None
        self.X_test_nb = None
        self.y_test_nb = None

        # --- TRANG BỊ CHO TAB KNN ---
        self.setup_tab(
            parent_tab=self.tab_knn,
            model_var="model_knn",
            x_test_var="X_test_knn",
            y_test_var="y_test_knn",
            predict_command=self.predict_knn,
            model_type="knn"
        )

        # --- TRANG BỊ CHO TAB HỒI QUY ---
        self.setup_tab(
            parent_tab=self.tab_reg,
            model_var="model_reg",
            x_test_var="X_test_reg",
            y_test_var="y_test_reg",
            predict_command=self.predict_reg,
            model_type="reg"
        )

        # --- TRANG BỊ CHO TAB NAIVE BAYES ---
        self.setup_tab(
            parent_tab=self.tab_nb,
            model_var="model_nb",
            x_test_var="X_test_nb",
            y_test_var="y_test_nb",
            predict_command=self.predict_nb,
            model_type="nb"
        )

    def setup_tab(self, parent_tab, model_var, x_test_var, y_test_var, predict_command, model_type):
        """Hàm dùng chung để tạo giao diện nút bấm và label cho từng Tab để code gọn gàng, đồng bộ"""
        
        # Frame chứa các nút tải file
        btn_frame = ctk.CTkFrame(parent_tab)
        btn_frame.pack(pady=20, fill="x", padx=40)

        # --- Nút Tải Mô Hình ---
        btn_load_model = ctk.CTkButton(
            btn_frame, 
            text="📁 Tải Mô hình (.joblib/.pkl)", 
            command=lambda: self.load_model(model_var, lbl_model_status, btn_predict),
            width=200, height=40, font=ctk.CTkFont(weight="bold")
        )
        btn_load_model.grid(row=0, column=0, padx=10, pady=10)
        
        lbl_model_status = ctk.CTkLabel(btn_frame, text="❌ Chưa tải", text_color="red")
        lbl_model_status.grid(row=0, column=1, padx=10, pady=10, sticky="w")

        # --- Nút Tải X_test ---
        btn_load_x = ctk.CTkButton(
            btn_frame, 
            text="📄 Tải X_test (.csv/.npy)", 
            command=lambda: self.load_csv(x_test_var, lbl_x_status, btn_predict),
            width=200, height=40, fg_color="#2b9348", hover_color="#007f5f", font=ctk.CTkFont(weight="bold")
        )
        btn_load_x.grid(row=1, column=0, padx=10, pady=10)
        
        lbl_x_status = ctk.CTkLabel(btn_frame, text="❌ Chưa tải", text_color="red")
        lbl_x_status.grid(row=1, column=1, padx=10, pady=10, sticky="w")

        # --- Nút Tải y_test ---
        btn_load_y = ctk.CTkButton(
            btn_frame, 
            text="🎯 Tải y_test (.csv/.npy)", 
            command=lambda: self.load_csv(y_test_var, lbl_y_status, btn_predict),
            width=200, height=40, fg_color="#b5179e", hover_color="#7209b7", font=ctk.CTkFont(weight="bold")
        )
        btn_load_y.grid(row=2, column=0, padx=10, pady=10)
        
        lbl_y_status = ctk.CTkLabel(btn_frame, text="❌ Chưa tải", text_color="red")
        lbl_y_status.grid(row=2, column=1, padx=10, pady=10, sticky="w")

        # --- Nút Dự Đoán ---
        btn_predict = ctk.CTkButton(
            parent_tab, 
            text="🚀 THỰC HIỆN DỰ ĐOÁN", 
            command=predict_command,
            width=250, height=50, 
            font=ctk.CTkFont(size=16, weight="bold"),
            state="disabled",
            fg_color="#fca311", hover_color="#e5e5e5", text_color="black"
        )
        btn_predict.pack(pady=15)

        # --- Khu vực hiển thị kết quả ---
        result_frame = ctk.CTkFrame(parent_tab, fg_color="#1e1e1e", corner_radius=15)
        result_frame.pack(fill="both", expand=True, padx=40, pady=(0, 20))
        
        lbl_result = ctk.CTkLabel(
            result_frame, 
            text="KẾT QUẢ SẼ HIỂN THỊ TẠI ĐÂY", 
            font=ctk.CTkFont(size=20),
            justify="center",
            pady=20
        )
        lbl_result.pack(expand=True, fill="both")
        
        # Lưu label kết quả vào biến của class để dễ cập nhật
        if model_type == "knn":
            self.lbl_result_knn = lbl_result
            self.btn_predict_knn = btn_predict
        elif model_type == "reg":
            self.lbl_result_reg = lbl_result
            self.btn_predict_reg = btn_predict
        elif model_type == "nb":
            self.lbl_result_nb = lbl_result
            self.btn_predict_nb = btn_predict


    def load_model(self, model_var_name, status_label, predict_button):
        filepath = filedialog.askopenfilename(title="Chọn file mô hình", filetypes=[("Model Files", "*.joblib *.pkl"), ("Joblib Files", "*.joblib"), ("Pickle Files", "*.pkl"), ("All Files", "*.*")])
        if filepath:
            try:
                model = joblib.load(filepath)
                setattr(self, model_var_name, model)
                filename = filepath.split('/')[-1]
                status_label.configure(text=f"✅ {filename}", text_color="#00b4d8")
                self.check_ready(predict_button, model_var_name)
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể tải mô hình:\n{str(e)}")

    def load_csv(self, data_var_name, status_label, predict_button):
        filepath = filedialog.askopenfilename(title="Chọn file dữ liệu", filetypes=[("Data Files", "*.csv *.npy"), ("CSV Files", "*.csv"), ("Numpy Files", "*.npy"), ("All Files", "*.*")])
        if filepath:
            try:
                if filepath.endswith('.csv'):
                    df = pd.read_csv(filepath)
                elif filepath.endswith('.npy'):
                    arr = np.load(filepath)
                    if len(arr.shape) == 1:
                        df = pd.DataFrame(arr, columns=['Target'])
                    else:
                        df = pd.DataFrame(arr)
                else:
                    messagebox.showerror("Lỗi", "Định dạng file không được hỗ trợ!")
                    return

                setattr(self, data_var_name, df)
                filename = filepath.split('/')[-1]
                status_label.configure(text=f"✅ {filename} ({df.shape[0]} dòng)", text_color="#00b4d8")
                
                # Check target for model regression or knn
                if "reg" in data_var_name:
                    self.check_ready(self.btn_predict_reg, "model_reg")
                elif "knn" in data_var_name:
                    self.check_ready(self.btn_predict_knn, "model_knn")
                elif "nb" in data_var_name:
                    self.check_ready(self.btn_predict_nb, "model_nb")
                    
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể tải tệp dữ liệu:\n{str(e)}")

    def check_ready(self, btn_predict, model_var_name):
        """Kiểm tra xem đã load đủ file cho Tab hiện tại chưa để mở khoá nút dự đoán"""
        if model_var_name == "model_knn":
            if self.model_knn is not None and self.X_test_knn is not None and self.y_test_knn is not None:
                btn_predict.configure(state="normal")
        elif model_var_name == "model_reg":
            if self.model_reg is not None and self.X_test_reg is not None and self.y_test_reg is not None:
                btn_predict.configure(state="normal")
        elif model_var_name == "model_nb":
            if self.model_nb is not None and self.X_test_nb is not None and self.y_test_nb is not None:
                btn_predict.configure(state="normal")

    def predict_knn(self):
        try:
            # Lấy data
            X_test = self.X_test_knn
            # Giả sử y_test chỉ có 1 cột
            y_test = self.y_test_knn.iloc[:, 0] if len(self.y_test_knn.columns) == 1 else self.y_test_knn
            
            # Dự đoán
            y_pred = self.model_knn.predict(X_test)
            
            # Tính toán Metric KNN (Phân loại)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            result_text = (
                f"ĐÁNH GIÁ MÔ HÌNH KNN (PHÂN LOẠI)\n\n"
                f"🎯 Accuracy (Độ chính xác): {acc:.4f}\n\n"
                f"💎 Precision (Độ chuẩn xác): {prec:.4f}\n\n"
                f"🔍 Recall (Độ phủ): {rec:.4f}\n\n"
                f"⭐ F1-Score: {f1:.4f}"
            )
            self.lbl_result_knn.configure(text=result_text, text_color="#00f5d4", font=ctk.CTkFont(size=22, weight="bold"))
            
        except Exception as e:
            messagebox.showerror("Lỗi Dự đoán KNN", f"Vui lòng kiểm tra lại cấu trúc file dữ liệu X_test và y_test.\nChi tiết lỗi:\n{str(e)}")

    def predict_reg(self):
        try:
            # Lấy data
            X_test = self.X_test_reg
            # Giả sử y_test chỉ có 1 cột
            y_test = self.y_test_reg.iloc[:, 0] if len(self.y_test_reg.columns) == 1 else self.y_test_reg
            
            # Dự đoán
            y_pred = self.model_reg.predict(X_test)
            
            # Tính toán Metric Hồi quy
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            result_text = (
                f"ĐÁNH GIÁ MÔ HÌNH HỒI QUY\n\n"
                f"📉 MAE (Sai số tuyệt đối): {mae:.4f}\n\n"
                f"📊 MSE (Sai số bình phương): {mse:.4f}\n\n"
                f"📏 RMSE (Căn bậc 2 MSE): {rmse:.4f}\n\n"
                f"🎯 R2 Score: {r2:.4f}"
            )
            self.lbl_result_reg.configure(text=result_text, text_color="#fee440", font=ctk.CTkFont(size=22, weight="bold"))
            
        except Exception as e:
            messagebox.showerror("Lỗi Dự đoán Hồi quy", f"Vui lòng kiểm tra lại cấu trúc file dữ liệu X_test và y_test.\nChi tiết lỗi:\n{str(e)}")

    def predict_nb(self):
        try:
            # Lấy data
            X_test = self.X_test_nb
            # Giả sử y_test chỉ có 1 cột
            y_test = self.y_test_nb.iloc[:, 0] if len(self.y_test_nb.columns) == 1 else self.y_test_nb
            
            # Dự đoán
            y_pred = self.model_nb.predict(X_test)
            
            # Tính toán Metric Naive Bayes (Phân loại)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            result_text = (
                f"ĐÁNH GIÁ MÔ HÌNH NAÏVE BAYES (PHÂN LOẠI)\n\n"
                f"🎯 Accuracy (Độ chính xác): {acc:.4f}\n\n"
                f"💎 Precision (Độ chuẩn xác): {prec:.4f}\n\n"
                f"🔍 Recall (Độ phủ): {rec:.4f}\n\n"
                f"⭐ F1-Score: {f1:.4f}"
            )
            self.lbl_result_nb.configure(text=result_text, text_color="#f15bb5", font=ctk.CTkFont(size=22, weight="bold"))
            
        except Exception as e:
            messagebox.showerror("Lỗi Dự đoán Naïve Bayes", f"Vui lòng kiểm tra lại cấu trúc file dữ liệu X_test và y_test.\nChi tiết lỗi:\n{str(e)}")


if __name__ == "__main__":
    app = PremiumPredictorApp()
    app.mainloop()
