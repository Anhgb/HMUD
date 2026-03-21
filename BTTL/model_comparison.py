# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import sys
import codecs

if sys.stdout.encoding != 'utf-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
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

# Bỏ qua các cảnh báo (đặc biệt là cảnh báo hội tụ của MLPClassifier cho output sạch)
warnings.filterwarnings('ignore')

def load_data():
    datasets = {}
    
    # 1. Tập dữ liệu Iris
    print("Loading Iris dataset...")
    iris = load_iris()
    datasets['Iris'] = (iris.data, iris.target)
    
    # 2. Tập dữ liệu Wine
    print("Loading Wine dataset...")
    wine = load_wine()
    datasets['Wine'] = (wine.data, wine.target)
    
    # 3. Tập dữ liệu Digits
    print("Loading Digits dataset...")
    digits = load_digits()
    datasets['Digits'] = (digits.data, digits.target)
    
    # 4. Tập dữ liệu Diabetes
    # Scikit-learn có 1 tập diabetes nhưng dùng cho bài toán Regression.
    # Để Classification (theo yêu cầu Naive Bayes, SVM, KNN), ta sẽ tải tập Pima Indians Diabetes
    print("Loading Diabetes classification dataset from OpenML...")
    try:
        diabetes = fetch_openml(name='diabetes', version=1, as_frame=False, parser='auto')
        datasets['Diabetes'] = (diabetes.data, diabetes.target)
    except Exception as e:
        print(f"Error loading Diabetes dataset: {e}")
        
    return datasets

def get_models():
    # Khởi tạo các mô hình phân lớp theo yêu cầu
    models = {
        'k-NN (k=3)': KNeighborsClassifier(n_neighbors=3),
        'k-NN (k=5)': KNeighborsClassifier(n_neighbors=5),
        'k-NN (k=7)': KNeighborsClassifier(n_neighbors=7),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Naive Bayes': GaussianNB(),
        'SVM': SVC(random_state=42),
        'ANN': MLPClassifier(random_state=42, max_iter=2000, hidden_layer_sizes=(100,))
    }
    return models

def evaluate_models(datasets, models):
    results_dict = {}
    
    for ds_name, (X, y) in datasets.items():
        print(f"\n--- Đang huấn luyện cho tập dữ liệu: {ds_name} ---")
        
        # Tiền xử lý dữ liệu: Label encode cho dạng y-target nếu là object/string (cần cho tập diabetes)
        le = LabelEncoder()
        y = le.fit_transform(y)
        
        # Tiền xử lý dữ liệu: Handle Missing Values (nếu có, ví dụ OpenML hay có np.nan)
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
        
        # Split train, test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        # Chuẩn hoá dữ liệu tốt cho SVM, KNN, ANN
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        ds_results = {}
        for model_name, model in models.items():
            print(f"   + Huấn luyện {model_name}...")
            # Huấn luyện
            model.fit(X_train_scaled, y_train)
            # Dự đoán
            y_pred = model.predict(X_test_scaled)
            # Đánh giá Accuracy
            acc = accuracy_score(y_test, y_pred)
            ds_results[model_name] = acc
            
        results_dict[ds_name] = ds_results
        
    return pd.DataFrame(results_dict)

def visualize_results(df_results):
    # Lập bảng so sánh (Console)
    print("\n" + "="*60)
    print("BẢNG SO SÁNH HIỆU NĂNG (ĐỘ CHÍNH XÁC - ACCURACY) \nGIỮA CÁC MÔ HÌNH VÀ TẬP DỮ LIỆU TRÊN TẬP TEST")
    print("="*60)
    # Hiển thị đẹp dưới dạng bảng pandas
    print(df_results.round(4).to_string())
    print("="*60)
    
    # Vẽ Bản đồ nhiệt (Heatmap)
    plt.figure(figsize=(10, 6))
    
    # Vẽ heatmap bằng seaborn, .T dùng để xoay ngang dọc nhìn cho dễ
    sns.heatmap(df_results, annot=True, cmap='coolwarm', fmt='.4f', 
                linewidths=.5, cbar_kws={'label': 'Độ chính xác (Accuracy)'})
    
    plt.title('Bản đồ nhiệt: So sánh hiệu năng các mô hình trên các tập dữ liệu', fontsize=14, pad=15)
    plt.ylabel('Các mô hình phân lớp', fontsize=12)
    plt.xlabel('Tập dữ liệu', fontsize=12)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("=== CHƯƠNG TRÌNH SO SÁNH HIỆU NĂNG MÔ HÌNH PHÂN LỚP ===")
    datasets = load_data()
    models = get_models()
    
    # Đánh giá và lấy bảng kết quả
    df_results = evaluate_models(datasets, models)
    
    # Vẽ biểu đồ nhiệt và in ra console bảng kết quả
    visualize_results(df_results)
