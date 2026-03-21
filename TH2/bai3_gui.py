import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
import numpy as np

class VectorMatrixApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chuyển đổi Vector và Ma trận")
        self.root.geometry("700x600")
        self.root.configure(bg="#f0f0f0") # Màu nền nhẹ

        # Cấu hình style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TLabel", background="#f0f0f0", font=("Segoe UI", 11))
        style.configure("TButton", font=("Segoe UI", 11, "bold"), padding=6)
        style.configure("TFrame", background="#f0f0f0")
        style.configure("Header.TLabel", font=("Segoe UI", 16, "bold"), foreground="#333")

        # Header
        header_frame = ttk.Frame(root)
        header_frame.pack(pady=20)
        ttk.Label(header_frame, text="CHUYỂN ĐỔI VECTOR ⇄ MA TRẬN", style="Header.TLabel").pack()

        # Frame nhập liệu
        input_frame = ttk.LabelFrame(root, text="Thông số đầu vào", padding=(20, 10))
        input_frame.pack(padx=20, pady=10, fill="x")

        # Grid layout cho input
        ttk.Label(input_frame, text="Số hàng (m):").grid(row=0, column=0, padx=10, pady=10, sticky="e")
        self.entry_m = ttk.Entry(input_frame, width=10, font=("Segoe UI", 11))
        self.entry_m.grid(row=0, column=1, padx=10, pady=10, sticky="w")

        ttk.Label(input_frame, text="Số cột (n):").grid(row=0, column=2, padx=10, pady=10, sticky="e")
        self.entry_n = ttk.Entry(input_frame, width=10, font=("Segoe UI", 11))
        self.entry_n.grid(row=0, column=3, padx=10, pady=10, sticky="w")

        # Nút thực hiện
        btn_frame = ttk.Frame(root)
        btn_frame.pack(pady=10)
        self.btn_calc = ttk.Button(btn_frame, text="🚀 Thực hiện chuyển đổi", command=self.process)
        self.btn_calc.pack(side="left", padx=10)
        
        ttk.Button(btn_frame, text="🧹 Xóa màn hình", command=self.clear_output).pack(side="left", padx=10)

        # Khu vực hiển thị kết quả
        result_frame = ttk.LabelFrame(root, text="Kết quả chi tiết", padding=(10, 10))
        result_frame.pack(padx=20, pady=10, fill="both", expand=True)

        self.output_text = scrolledtext.ScrolledText(result_frame, width=80, height=20, font=("Consolas", 10), state="normal")
        self.output_text.pack(fill="both", expand=True)

    def log(self, message, tag=None):
        self.output_text.insert(tk.END, message + "\n", tag)
        self.output_text.see(tk.END)

    def clear_output(self):
        self.output_text.delete(1.0, tk.END)

    def process(self):
        self.clear_output()
        
        try:
            m_str = self.entry_m.get()
            n_str = self.entry_n.get()

            if not m_str or not n_str:
                messagebox.showwarning("Thiếu thông tin", "Vui lòng nhập đầy đủ giá trị m và n!")
                return
            
            m = int(m_str)
            n = int(n_str)
            
            if m <= 0 or n <= 0:
                messagebox.showerror("Lỗi giá trị", "m và n phải là số nguyên dương!")
                return

        except ValueError:
            messagebox.showerror("Lỗi định dạng", "Vui lòng kiểm tra lại m và n (phải là số nguyên)!")
            return

        k = m * n
        
        # Định dạng text output
        self.output_text.tag_config("title", foreground="blue", font=("Consolas", 11, "bold"))
        self.output_text.tag_config("success", foreground="green", font=("Consolas", 10, "bold"))

        self.log(f"=== BẮT ĐẦU XỬ LÝ (m={m}, n={n}, k={k}) ===", "title")
        self.log("-" * 60)

        # 1. Tạo vector ngẫu nhiên x
        self.log(f"1. Tạo vector ngẫu nhiên x thuộc R^{k}:")
        x = np.random.rand(k)
        # Làm tròn để hiển thị đẹp hơn
        x_rounded = np.round(x, 4)
        self.log(str(x_rounded))
        self.log("")

        # Chuyển x -> Ma trận X
        self.log(f"2. Chuyển x thành ma trận X ({m}x{n}):", "title")
        X = x.reshape(m, n)
        X_rounded = np.round(X, 4)
        self.log(str(X_rounded))
        self.log("")

        # 2. Chuyển X -> vector x_new
        self.log(f"3. Chuyển ma trận X trở lại thành vector:", "title")
        x_new = X.flatten()
        x_new_rounded = np.round(x_new, 4)
        self.log(str(x_new_rounded))
        self.log("")

        # Kiểm tra
        are_equal = np.allclose(x, x_new)
        self.log("-" * 60)
        start_msg = "KẾT QUẢ KIỂM TRA: "
        result_msg = "Trùng khớp (Thành công)" if are_equal else "Không trùng khớp (Thất bại)"
        self.log(start_msg + result_msg, "success" if are_equal else None)

if __name__ == "__main__":
    root = tk.Tk()
    app = VectorMatrixApp(root)
    root.mainloop()
