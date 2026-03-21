
import numpy as np

def main():
    # 1. Tạo các vector
    # a) Vector 10 chiều, giá trị số nguyên ngẫu nhiên từ 1 đến 10
    # randint(low, high, size): high là exclusive nên dùng 11 để lấy được 10
    vector1 = np.random.randint(1, 11, 10)

    # b) Vector 20 chiều, giá trị số thực [-1, 1]
    vector2 = np.random.uniform(-1, 1, 20)

    # c) Vector ngẫu nhiên có số chiều và khoảng giá trị nhập từ bàn phím
    print("--- Nhập thông tin cho Vector 3 ---")
    try:
        n = int(input("Nhập số chiều (n): "))
        min_val = float(input("Nhập giá trị tối thiểu: "))
        max_val = float(input("Nhập giá trị tối đa: "))
        
        # Tạo vector 3 với giá trị thực ngẫu nhiên trong khoảng [min, max]
        vector3 = np.random.uniform(min_val, max_val, n)
    except ValueError:
        print("Lỗi: Vui lòng nhập số hợp lệ!")
        return

    # 2. In các vector ra màn hình
    print("\n--- Kết quả ---")
    print("Vector 1 (10 số nguyên [1, 10]):")
    print(vector1)
    
    print("\nVector 2 (20 số thực [-1, 1]):")
    print(vector2)
    
    print("\nVector 3 (Người dùng nhập):")
    print(vector3)

    # 3. Xuất các vector ra file .txt
    output_filename = "vectors.txt"
    try:
        # Sử dụng 'w' để ghi đè hoặc tạo mới
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write("Vector 1 (Integers 1-10):\n")
            # fmt='%d' để lưu số nguyên
            np.savetxt(f, vector1.reshape(1, -1), fmt='%d', delimiter=', ')
            
            f.write("\nVector 2 (Floats -1 to 1):\n")
            # fmt='%.4f' để lưu số thực với 4 chữ số thập phân
            np.savetxt(f, vector2.reshape(1, -1), fmt='%.4f', delimiter=', ')
            
            f.write("\nVector 3 (Custom):\n")
            np.savetxt(f, vector3.reshape(1, -1), fmt='%.4f', delimiter=', ')
            
        print(f"\nĐã xuất dữ liệu ra file '{output_filename}' thành công.")
    except Exception as e:
        print(f"\nCó lỗi khi ghi file: {e}")

if __name__ == "__main__":
    main()
