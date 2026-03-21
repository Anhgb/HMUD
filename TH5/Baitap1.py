import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Nhập các mô hình từ scikit-learn
from sklearn.linear_model import (
    LinearRegression,  # Mô hình OLS
    Ridge,             # Mô hình Ridge
    Lasso,             # Mô hình Lasso
    BayesianRidge,     # Mô hình Bayesian
    TweedieRegressor   # Mô hình GLM (Generalized Linear Model)
)

def evaluate_model(name, y_true, y_pred):
    """Hàm in kết quả đánh giá mô hình"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print(f"| {name:<25} | {mae:<10.4f} | {mse:<10.4f} | {rmse:<10.4f} | {r2:<10.4f} |")
    return [name, mae, mse, rmse, r2]

def main():
    # 1. TẢI VÀ CHUẨN BỊ DỮ LIỆU
    print("-" * 75)
    print("Dang tai tap du lieu California Housing...")
    california = fetch_california_housing()
    X = pd.DataFrame(california.data, columns=california.feature_names)
    y = pd.Series(california.target)
    
    # Chia tập dữ liệu (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Chuẩn hóa dữ liệu (Z-score normalization)
    # StandardScaler giúp các mô hình (đặc biệt là Ridge, Lasso, Poly) hội tụ tốt hơn
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n--- BANG KET QUA DANH GIA CAC MO HINH HOI QUY ---")
    print("-" * 75)
    print(f"| {'Ten Mo Hinh':<25} | {'MAE':<10} | {'MSE':<10} | {'RMSE':<10} | {'R2':<10} |")
    print("-" * 75)

    results = []

    # ---------------------------------------------------------
    # 1. Mô hình OLS (Ordinary Least Squares) - Linear Regression
    # ---------------------------------------------------------
    ols_model = LinearRegression()
    ols_model.fit(X_train_scaled, y_train)
    y_pred_ols = ols_model.predict(X_test_scaled)
    results.append(evaluate_model("OLS (Linear Regression)", y_test, y_pred_ols))

    # ---------------------------------------------------------
    # 2. Mô hình Ridge Regression (Hồi quy có kiểm soát L2)
    # ---------------------------------------------------------
    ridge_model = Ridge(alpha=1.0) # alpha là tham số điều chuẩn
    ridge_model.fit(X_train_scaled, y_train)
    y_pred_ridge = ridge_model.predict(X_test_scaled)
    results.append(evaluate_model("Ridge Regression", y_test, y_pred_ridge))

    # ---------------------------------------------------------
    # 3. Mô hình Lasso Regression (Hồi quy có kiểm soát L1)
    # ---------------------------------------------------------
    lasso_model = Lasso(alpha=0.01) # Chọn alpha nhỏ để không bị triệt tiêu quá nhiều đặc trưng
    lasso_model.fit(X_train_scaled, y_train)
    y_pred_lasso = lasso_model.predict(X_test_scaled)
    results.append(evaluate_model("Lasso Regression", y_test, y_pred_lasso))

    # ---------------------------------------------------------
    # 4. Mô hình Bayesian Regression
    # ---------------------------------------------------------
    bayesian_model = BayesianRidge()
    bayesian_model.fit(X_train_scaled, y_train)
    y_pred_bayesian = bayesian_model.predict(X_test_scaled)
    results.append(evaluate_model("Bayesian Regression", y_test, y_pred_bayesian))

    # ---------------------------------------------------------
    # 5. Mô hình GLM (Generalized Linear Model)
    # Sử dụng TweedieRegressor trong sklearn (bản chất là mở rộng của phân phối exponential family)
    # power=0 tương đương với phân phối Gaussian (Linear)
    # ---------------------------------------------------------
    glm_model = TweedieRegressor(power=0, alpha=0.5, link='identity')
    glm_model.fit(X_train_scaled, y_train)
    y_pred_glm = glm_model.predict(X_test_scaled)
    results.append(evaluate_model("GLM (Tweedie Gaussian)", y_test, y_pred_glm))

    # ---------------------------------------------------------
    # 6. Mô hình Polynomial Regression (Hồi quy đa thức)
    # Lưu ý: Cần tạo đặc trưng đa thức trước (ở đây chọn bậc 2)
    # ---------------------------------------------------------
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    # Transform tập train và test
    X_train_poly = poly_features.fit_transform(X_train_scaled)
    X_test_poly = poly_features.transform(X_test_scaled)
    
    # Sử dụng Linear Regression (OLS) trên tập các đặc trưng đa thức mới này
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    y_pred_poly = poly_model.predict(X_test_poly)
    results.append(evaluate_model("Polynomial (d=2)", y_test, y_pred_poly))
    
    print("-" * 75)
    
    # ---------------------------------------------------------
    # VẼ BIỂU ĐỒ SO SÁNH
    # ---------------------------------------------------------
    df_results = pd.DataFrame(results, columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2'])
    
    plt.figure(figsize=(12, 6))
    
    # Biểu đồ so sánh RMSE (Sai số càng thấp càng tốt)
    plt.subplot(1, 2, 1)
    bars_rmse = plt.barh(df_results['Model'], df_results['RMSE'], color='skyblue')
    plt.xlabel('RMSE')
    plt.title('So Sanh Sai So RMSE (Cang thap cang tot)')
    plt.gca().invert_yaxis() # Đảo ngược trục y cho dễ nhìn
    # Gắn nhãn lên thanh
    for bar in bars_rmse:
        plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
                 f"{bar.get_width():.3f}", va='center')

    # Biểu đồ so sánh R2 (Hệ số bằng 1 là hoàn hảo)
    plt.subplot(1, 2, 2)
    bars_r2 = plt.barh(df_results['Model'], df_results['R2'], color='lightgreen')
    plt.xlabel('R-Squared (R2)')
    plt.title('So Sanh R2 (Cang gan 1 cang tot)')
    plt.gca().invert_yaxis()
    for bar in bars_r2:
        plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
                 f"{bar.get_width():.3f}", va='center')

    plt.tight_layout()
    # plt.show() # Tạm comment để chạy ngầm không bị kẹt Terminal. Bạn có thể mở comment này khi mở bên IDE.
    print("Da thuc hien xong danh gia tren tat ca mo hinh!")

if __name__ == "__main__":
    main()
