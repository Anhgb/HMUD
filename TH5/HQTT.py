import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

# Tải tập dữ liệu Boston từ OpenML vì hàm load_boston đã bị xóa khỏi sklearn trong các phiên bản mới
boston = fetch_openml(name='boston', version=1, parser='auto')
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.3, random_state=42)
print("Kích thước tập huấn luyện:", X_train.shape)
print("Kích thước tập kiểm thử:", X_test.shape)
from sklearn.linear_model import LinearRegression
# Kkhởi tạo mh hqtt
model = LinearRegression()
# hl mh vs tập dl train
model.fit(X_train, y_train)
# in tt về mô hình 
print("Hệ số chẵn:", model.intercept_)
print("Hệ số dốc:", model.coef_)