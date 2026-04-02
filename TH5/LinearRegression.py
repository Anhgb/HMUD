from sklearn.linear_model import LinearRegression
# Kkhởi tạo mh hqtt
model = LinearRegression()
# hl mh vs tập dl train
model.fit(X_train, y_train)
# in tt về mô hình 
print("Hệ số chẵn:", model.intercept_)
print("Hệ số dốc:", model.coef_)