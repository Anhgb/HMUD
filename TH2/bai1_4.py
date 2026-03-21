
import numpy as np

def main():
    print("--- Chương trình thực hiện các phép toán trên vector (R^n) ---\n")
    
    # 1. Tạo ngẫu nhiên 2 vector a, b thuộc R^n
    # Chọn n ngẫu nhiên từ 3 đến 8 (ví dụ)
    n = np.random.randint(3, 9)
    print(f"Số chiều n được chọn ngẫu nhiên: {n}")
    
    # Tạo vector a, b với giá trị ngẫu nhiên. Ví dụ số nguyên [-10, 10]
    a = np.random.randint(-10, 11, n)
    b = np.random.randint(-10, 11, n)
    
    print(f"Vector a: {a}")
    print(f"Vector b: {b}")
    print("-" * 30)

    # 2. Thực hiện các phép toán
    
    # a) Chuyển vị (Transpose)
    # Lưu ý: Trong numpy, mảng 1D (shape=(n,)) khi .T vẫn là chính nó.
    # Để thấy rõ chuyển vị, ta cần reshape thành ma trận hàng (1, n) hoặc cột (n, 1)
    print("\n1. Chuyển vị vector a:")
    # Coi a là vector hàng => chuyển vị thành vector cột
    a_col = a.reshape(-1, 1)
    print(f"Vector a dạng cột (chuyển vị):\n{a_col}")
    
    print("\nChuyển vị vector b:")
    b_col = b.reshape(-1, 1)
    print(f"Vector b dạng cột (chuyển vị):\n{b_col}")

    # b) Cộng 2 vector (a + b)
    print("\n2. Tổng hai vector (a + b):")
    # Phép cộng element-wise
    sum_ab = a + b
    print(f"{a} + {b} = {sum_ab}")

    # c) Nhân 2 vector (Element-wise multiplication)
    # Đề bài: "nhân 2 vector". Thường hiểu là nhân từng phần tử nếu phân biệt với "nhân vô hướng".
    print("\n3. Tích 2 vector (Element-wise - Hadamard product):")
    mul_ab = a * b
    print(f"{a} * {b} = {mul_ab}")

    # d) Nhân vô hướng 2 vector (Dot product)
    print("\n4. Tích vô hướng (Dot product):")
    dot_product = np.dot(a, b)
    # Cách khác: a @ b hoặc np.inner(a, b)
    print(f"<{a}, {b}> = {dot_product}")
    
    # Giải thích thêm
    print(f"Giải thích: {' + '.join([f'({x}*{y})' for x, y in zip(a, b)])} = {dot_product}")

if __name__ == "__main__":
    main()
