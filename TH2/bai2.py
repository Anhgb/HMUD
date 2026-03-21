import numpy as np

# 1. Tạo ngẫu nhiên 2 ma trận A thuộc R^m*n, B thuộc R^n*k
m, n, k = 3, 4, 2  # Kích thước tự chọn
print(f"Kích thước: m={m}, n={n}, k={k}")

# Tạo ma trận A (m x n) với các giá trị ngẫu nhiên từ 1 đến 10
A = np.random.randint(1, 11, size=(m, n))
print("\nMa trận A (m x n):")
print(A)

# Tạo ma trận B (n x k) với các giá trị ngẫu nhiên từ 1 đến 10
B = np.random.randint(1, 11, size=(n, k))
print("\nMa trận B (n x k):")
print(B)

# 2. Thực hiện phép chuyển vị với A, B
A_T = np.transpose(A) # Hoặc A.T
B_T = np.transpose(B) # Hoặc B.T
print("\nMa trận chuyển vị của A (A^T):")
print(A_T)
print("\nMa trận chuyển vị của B (B^T):")
print(B_T)

# 3. Nhân 1 đại lượng vô hướng u với A, B
u = 2 # Đại lượng vô hướng tự chọn
print(f"\nNhân đại lượng vô hướng u = {u} với A:")
print(u * A)
print(f"\nNhân đại lượng vô hướng u = {u} với B:")
print(u * B)

# 4. Tạo ma trận C thuộc R^m*n, C + A
# Tạo ma trận C cùng kích thước với A
C = np.random.randint(1, 11, size=(m, n))
print("\nMa trận C (m x n):")
print(C)

print("\nKết quả phép cộng C + A:")
print(C + A)

# 5. Thực hiện phép nhân ma trận: A * B
tich_AB = np.dot(A, B) # Hoặc A @ B
print("\nKết quả phép nhân ma trận A * B:")
print(tich_AB)
