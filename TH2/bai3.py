import numpy as np

def main():
    print("--- Bài 3: Chuyển đổi giữa Vector và Ma trận ---")
    
    # Nhập m và n từ người dùng
    try:
        m = int(input("Nhập m (số hàng): "))
        n = int(input("Nhập n (số cột): "))
    except ValueError:
        print("Lỗi: Vui lòng nhập số nguyên hợp lệ.")
        return

    k = m * n
    
    # 1. Tạo vector ngẫu nhiên x thuộc R^k -> chuyển thành ma trận X thuộc R^(m x n)
    print(f"\n1. Tạo vector ngẫu nhiên x có kích thước k = {m}*{n} = {k}")
    x = np.random.rand(k) # Vector ngẫu nhiên
    print(f"Vector x (10 phần tử đầu tiên nếu k lớn):\n{x[:10]} {'...' if k > 10 else ''}")
    
    print(f"\nChuyển x thành ma trận X kích thước {m}x{n}:")
    X = x.reshape(m, n)
    print("Ma trận X:")
    print(X)
    
    # 2. Chuyển ma trận X thuộc R^(m x n) thành vector x thuộc R^k
    print(f"\n2. Chuyển ma trận X trở lại thành vector x thuộc R^{k}")
    x_new = X.flatten() # Hoặc dùng X.reshape(-1)
    print(f"Vector x sau khi chuyển đổi (10 phần tử đầu tiên):\n{x_new[:10]} {'...' if k > 10 else ''}")
    
    # Kiểm tra tính toàn vẹn
    check = np.allclose(x, x_new)
    print(f"\nKiểm tra: Vector ban đầu và vector sau khi chuyển đổi có giống nhau không? {'Có' if check else 'Không'}")

if __name__ == "__main__":
    main()
