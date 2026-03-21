from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Tải tập dl và chia tỷ lệ 70-30
boston = fetch_openml(name='boston', version=1, parser='auto')
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.3, random_state=42)

# Khởi tạo mô hình
model = LinearRegression()
# huấn luyện với tập dữ liệu train
model.fit(X_train, y_train)
# in thông tin về mô hình 
print("He so chan:", model.intercept_)
print("He so doc:", model.coef_)