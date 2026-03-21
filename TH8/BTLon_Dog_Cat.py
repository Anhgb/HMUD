import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

def train_ann_with_cv(x_train_path, y_train_path):
    """
    Hàm huấn luyện mô hình mạng nơ-ron nhân tạo với 10-fold CV.
    
    Tham số:
    - x_train_path: Đường dẫn đến file numpy chứa ma trận đặc trưng (X_train.npy)
    - y_train_path: Đường dẫn đến file numpy chứa vector nhãn (y_train.npy)
    
    Đầu ra:
    - Mô hình ANN đã được huấn luyện.
    """
    
    # 1. Đọc tập dữ liệu train vào bộ nhớ
    print(f"Đang đọc dữ liệu từ '{x_train_path}' và '{y_train_path}'...")
    try:
        X_train = np.load(x_train_path)
        y_train = np.load(y_train_path)
        print(f"Kích thước X_train: {X_train.shape}, y_train: {y_train.shape}")
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")
        return None

    # 2. Tiến hành thực hiện MinMaxScaler trên tập thuộc tính X
    print("Đang thực hiện MinMaxScaler trên tập thuộc tính X...")
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)

    # 3. Khởi tạo mô hình mạng nơ ron nhân tạo (Artificial Neural Network)
    # Sử dụng Multi-Layer Perceptron với 1 lớp ẩn gồm 100 nơ-ron
    print("Đang khởi tạo mô hình mạng nơ-ron nhân tạo (ANN)...")
    ann_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)

    # 4. Huấn luyện và đánh giá mô hình với tập dữ liệu train theo thiết lập 10-fold CV
    print("Đang tiến hành đánh giá mô hình bằng 10-fold Cross-Validation...")
    # StratifiedKFold giúp giữ nguyên tỷ lệ nhãn lớp trong mỗi fold
    cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    # Thực hiện 10-fold CV (n_jobs=-1 giúp chạy đa luồng cho nhanh)
    cv_scores = cross_val_score(ann_model, X_train, y_train, cv=cv_strategy, scoring='accuracy', n_jobs=-1)
    
    print("-" * 30)
    print("KẾT QUẢ 10-FOLD CROSS-VALIDATION:")
    print(f"Độ chính xác của từng fold: {np.round(cv_scores * 100, 2)}%")
    print(f"Độ chính xác trung bình: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")
    print("-" * 30)

    # Tiến hành huấn luyện (fit) mô hình cuối cùng trên toàn bộ dữ liệu train
    # để mô hình học được nhiều thông tin nhất trước khi trả về.
    print("Đang huấn luyện mô hình trên toàn bộ tập dữ liệu train...")
    ann_model.fit(X_train, y_train)
    print("Huấn luyện hoàn tất!")

    # 5. Trả về mô hình đã được huấn luyện
    return ann_model
    

# ==========================================
# PHẦN CHẠY THỬ NGHIỆM (MAIN)
# ==========================================
if __name__ == "__main__":
    # Đường dẫn giả định đến file .npy đã được tạo ra từ bước tiền xử lý trước đó
    path_X = 'X_train.npy'
    path_y = 'y_train.npy'
    
    # Gọi hàm và nhận mô hình đầu ra
    trained_model = train_ann_with_cv(path_X, path_y)
    
    if trained_model is not None:
        print("\nĐã tạo thành công mô hình:")
        print(trained_model)