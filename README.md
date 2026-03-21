# HMUD-K22 Machine Learning & Image Processing

Repository này chứa các bài tập thực hành, bài tập lớn và dự án liên quan đến học máy (Machine Learning), xử lý ảnh (Image Processing) và xây dựng giao diện người dùng (GUI) với Tkinter bằng Python.

## Cấu trúc thư mục

Dự án được tổ chức theo các thư mục bài thực hành (TH) và bài tập lớn:

*   **`TH1` - `TH9`**: Trình tự các bài thực hành trên lớp. Các chủ đề bao gồm:
    *   Thao tác cơ bản với ma trận, vector bằng `NumPy`.
    *   Xử lý ảnh cơ bản với `OpenCV` (đọc ảnh, chuyển đổi ảnh màu sang xám, ...).
    *   Xây dựng giao diện ứng dụng desktop bằng `Tkinter`.
    *   Huấn luyện các mô hình Machine Learning cổ điển: Hồi quy tuyến tính (Linear Regression), SVM, ...
    *   Phân loại ảnh sử dụng Multi-layer Perceptron (MLP) / Artificial Neural Network (ANN).
*   **`CT_DESKTOP`**: Các chương trình ứng dụng desktop hoàn chỉnh tích hợp giao diện Tkinter và các thuật toán học máy.
*   **`BTTL`**: Bài tập tự làm.
*   **`Image/`**: Thư mục chứa các ảnh dùng để test và xử lý trong các bài tập.
*   Các file dữ liệu:
    *   `.csv`: Tập dữ liệu dạng bảng.
    *   `.npy`: Dữ liệu vector/ma trận đặc trưng ảnh đã được trích xuất (X_train, y_train, X_test, y_test).
    *   `.pkl`, `.h5`: Các mô hình Machine Learning và Deep Learning (ví dụ: mô hình nhận diện chó mèo) đã được huấn luyện và lưu lại để sử dụng.

## Công nghệ sử dụng

*   **Python 3.x**
*   **Tkinter / CustomTkinter**: Xây dựng giao diện người dùng (GUI).
*   **NumPy & Pandas**: Xử lý dữ liệu, thao tác với mảng đa chiều.
*   **OpenCV (`cv2`) & PIL (Pillow)**: Đọc, hiển thị và xử lý các phép toán trên hình ảnh.
*   **Scikit-Learn (`sklearn`)**: Triển khai các thuật toán Machine Learning truyền thống (SVM, Linear Regression...).
*   **TensorFlow / Keras**: Xây dựng và huấn luyện các mô hình mạng nơ-ron nhân tạo (ANN).
*   **Matplotlib**: Vẽ biểu đồ trực quan hóa dữ liệu và kết quả huấn luyện.

## Hướng dẫn sử dụng

1.  **Clone repository** về máy:
    ```bash
    git clone <URL_CUA_REPO>
    cd HMUD-K22
    ```
2.  **Cài đặt các thư viện** cần thiết:
    Bạn có thể cài đặt thông qua `pip`:
    ```bash
    pip install numpy pandas opencv-python pillow scikit-learn tensorflow matplotlib
    ```
    *(Nếu dự án có file `requirements.txt`, chạy lệnh: `pip install -r requirements.txt`)*
3.  **Chạy các chương trình**:
    Sử dụng Python để chạy trực tiếp các file `.py` trong từng thư mục. Ví dụ:
    ```bash
    python TH9/APP_BaiTap.py
    ```

## Tác giả / Nguồn
Code được phát triển trong quá trình học tập và làm bài tập thực hành môn học.
