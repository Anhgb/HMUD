import pandas as pd
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

# Giả sử chúng ta có hai cột features và cột label
# Tạo dữ liệu giả lập để không bị lỗi df None
data = {
    'car': ['Toyota', 'Honda', 'Ford', 'BMW', 'Audi', 'Honda', 'Toyota', 'BMW', 'Audi', 'Ford'],
    'country': ['Japan', 'Japan', 'USA', 'Germany', 'Germany', 'Japan', 'Japan', 'Germany', 'Germany', 'USA'],
    'label': ['A', 'B', 'A', 'C', 'B', 'B', 'A', 'C', 'B', 'A']
}
df = pd.DataFrame(data)

X = df[['car', 'country']]
y = df['label']

# Mã hóa cột nhãn y
label_encoder = LabelEncoder()

# Chia dữ liệu trước khi mã hóa y
X_train, X_test, y_train_raw, y_test_raw = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit LabelEncoder với y_train_raw và transform tập y_train_raw
y_train_encoded = label_encoder.fit_transform(y_train_raw)

# Chỉ transform tập y_test_raw
y_test_encoded = label_encoder.transform(y_test_raw)

print(f"Nhãn huấn luyện nguyên thủy: \n{y_train_raw.values}")
print(f"Nhãn huấn luyện đã được mã hóa: {y_train_encoded}")
print(f"\nNhãn kiểm tra nguyên thủy: \n{y_test_raw.values}")
print(f"Nhãn kiểm tra đã được mã hóa: {y_test_encoded}")

# Bây giờ có thể dùng y_train_encoded và y_test_encoded để huấn luyện mô hình 
# Giả định rằng các features cũng được xử lý (ví dụ One-Hot Encoding)
