import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import tkinter as tk
from tkinter import ttk, messagebox
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sklearn.datasets import load_iris, load_wine, load_digits, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# Bỏ qua các cảnh báo (ví dụ cảnh báo hội tụ của ANN để output gọn gàng)
warnings.filterwarnings('ignore')

def load_datasets():
    """Tải các tập dữ liệu yêu cầu."""
    datasets = {}
    
    # 1. Iris
    iris = load_iris()
    datasets['Iris'] = (iris.data, iris.target)
    
    # 2. Wine
    wine = load_wine()
    datasets['Wine'] = (wine.data, wine.target)
    
    # 3. Digits
    digits = load_digits()
    datasets['Digits'] = (digits.data, digits.target)
    
    # 4. Diabetes (Pima Indians Diabetes - Phân lớp)
    print("Đang tải dữ liệu Diabetes từ OpenML (có thể mất vài giây)...")
    try:
        diabetes = fetch_openml(name='diabetes', version=1, as_frame=False, parser='auto')
        datasets['Diabetes'] = (diabetes.data, diabetes.target)
    except Exception as e:
        print(f"Lỗi khi tải Diabetes: {e}")
        
    return datasets

def get_models():
    """Khởi tạo các mô hình phân lớp."""
    return {
        'k-NN (k=3)': KNeighborsClassifier(n_neighbors=3),
        'k-NN (k=5)': KNeighborsClassifier(n_neighbors=5),
        'k-NN (k=7)': KNeighborsClassifier(n_neighbors=7),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Naive Bayes': GaussianNB(),
        'SVM': SVC(random_state=42),
        'ANN': MLPClassifier(random_state=42, max_iter=2000, hidden_layer_sizes=(100,))
    }

class MachineLearningApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Phần Mềm Đánh Giá Mô Hình Phân Lớp")
        self.geometry("1000x800")
        
        # --- Khung chứa các nút điều khiển ---
        control_frame = ttk.Frame(self)
        control_frame.pack(pady=10, fill=tk.X)
        
        self.btn_run = ttk.Button(control_frame, text="Bắt đầu Tải dữ liệu & Huấn luyện", command=self.start_training)
        self.btn_run.pack(side=tk.TOP, pady=5)
        
        self.status_var = tk.StringVar()
        self.status_var.set("Sẵn sàng.")
        self.status_label = ttk.Label(control_frame, textvariable=self.status_var, font=("Arial", 10, "italic"))
        self.status_label.pack(side=tk.TOP)

        # --- Khung chứa bảng dữ liệu (Treeview) ---
        self.table_frame = ttk.Frame(self)
        self.table_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=False)
        
        # Khởi tạo Treeview rỗng
        self.tree = ttk.Treeview(self.table_frame, height=8)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Thanh cuộn cho bảng
        scrollbar = ttk.Scrollbar(self.table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=scrollbar.set)

        # --- Khung chứa biểu đồ (Matplotlib Canvas) ---
        self.plot_frame = ttk.Frame(self)
        self.plot_frame.pack(pady=10, fill=tk.BOTH, expand=True)

    def start_training(self):
        self.btn_run.config(state=tk.DISABLED)
        self.status_var.set("Đang tải dữ liệu và huấn luyện mô hình... Vui lòng đợi (có thể mất vài chục giây)!")
        
        # Chạy trong một luồng (thread) riêng để không làm đơ giao diện
        thread = threading.Thread(target=self.run_evaluation)
        thread.daemon = True
        thread.start()

    def run_evaluation(self):
        try:
            datasets = load_datasets()
            models = get_models()
            results = {}

            for ds_name, (X, y) in datasets.items():
                self.status_var.set(f"Đang huấn luyện mô hình cho tập dữ liệu: {ds_name}...")
                
                le = LabelEncoder()
                y = le.fit_transform(y)
                
                imputer = SimpleImputer(strategy='mean')
                X = imputer.fit_transform(X)
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                ds_results = {}
                for model_name, model in models.items():
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    acc = accuracy_score(y_test, y_pred)
                    ds_results[model_name] = acc
                    
                results[ds_name] = ds_results

            df_results = pd.DataFrame(results)
            
            # Cập nhật giao diện từ luồng chính (main thread)
            self.after(0, self.update_gui, df_results)
            
        except Exception as e:
            self.after(0, self.show_error, str(e))

    def update_gui(self, df_results):
        self.status_var.set("Hoàn tất huấn luyện!")
        self.btn_run.config(state=tk.NORMAL)
        
        # 1. Cập nhật Bảng (Treeview)
        self.tree.delete(*self.tree.get_children()) # Xóa dữ liệu cũ
        
        columns = ["Mô hình"] + list(df_results.columns)
        self.tree["columns"] = columns
        self.tree["show"] = "headings"
        
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor=tk.CENTER, width=120)
            
        for model_name, row in df_results.iterrows():
            values = [model_name] + [f"{val:.4f}" for val in row.values]
            self.tree.insert("", tk.END, values=values)

        # 2. Cập nhật Biểu đồ (Heatmap)
        for widget in self.plot_frame.winfo_children():
            widget.destroy() # Xóa biểu đồ cũ
            
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(df_results, annot=True, cmap='RdYlGn', fmt='.2%', linewidths=.5, cbar_kws={'label': 'Độ chính xác (Accuracy)'}, ax=ax)
        ax.set_title('Bản đồ nhiệt: So sánh hiệu năng các mô hình trên các tập dữ liệu', fontsize=12, pad=10)
        ax.set_ylabel('Các mô hình phân lớp', fontsize=10)
        ax.set_xlabel('Tập dữ liệu', fontsize=10)
        fig.tight_layout()
        
        # Nhúng Matplotlib vào Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def show_error(self, error_msg):
        self.status_var.set("Đã xảy ra lỗi!")
        self.btn_run.config(state=tk.NORMAL)
        messagebox.showerror("Lỗi", f"Đã xảy ra lỗi trong quá trình chạy:\n{error_msg}")

if __name__ == "__main__":
    app = MachineLearningApp()
    app.mainloop()