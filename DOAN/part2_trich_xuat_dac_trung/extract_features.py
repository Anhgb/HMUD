# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
=============================================================================
MODULE: extract_features.py
MÔ TẢ: Script trích xuất đặc trưng Hand Landmarks từ video bằng MediaPipe.
        Mỗi video được đọc frame-by-frame, trích xuất tọa độ (x, y, z) của
        21 điểm khớp bàn tay (21 landmarks x 3 tọa độ = 63 số/tay).
        Dùng 2 tay -> 126 số/frame.
        
        Chuỗi N frame -> sequence shape: (N, 126)
        Padding/Truncating về SEQUENCE_LENGTH frame cố định.
        
OUTPUT: 
        dataset/X_sequences.npy  -> shape: (num_samples, SEQUENCE_LENGTH, 126)
        dataset/y_labels.npy     -> shape: (num_samples,)
        dataset/label_names.npy  -> mảng tên các nhãn
=============================================================================
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import sys
from pathlib import Path

# Thêm thư mục cha vào path để import parse_label_studio
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from parse_label_studio import quet_thu_muc_video, LABEL_MAP


# ============================================================
#  HYPERPARAMETERS - CẦN ĐIỀU CHỈNH
# ============================================================
# Số frame tối đa cho mỗi chuỗi (sequence)
# Ký hiệu 2-3 giây @ 30fps -> ~60-90 frame; chọn 60 là hợp lý
SEQUENCE_LENGTH = 60

# Số tọa độ mỗi tay (21 landmarks x 3 tọa độ x,y,z)
NUM_FEATURES_PER_HAND = 21 * 3  # = 63

# Số tay tối đa phát hiện (2 tay -> 126 features/frame)
MAX_HANDS = 2
TOTAL_FEATURES = NUM_FEATURES_PER_HAND * MAX_HANDS  # = 126

# Cấu hình MediaPipe Hands
MEDIAPIPE_CONFIDENCE = 0.5  # Ngưỡng độ tin cậy phát hiện tay

# Đường dẫn
VIDEO_ROOT = "part1_thu_thap_du_lieu/raw_videos"
OUTPUT_DIR = "part2_trich_xuat_dac_trung/dataset"


# ============================================================
#  KHỞI TẠO MEDIAPIPE
# ============================================================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


class TrichXuatLandmark:
    """
    Lớp đóng gói toàn bộ logic trích xuất hand landmarks từ video.
    """

    def __init__(self, sequence_length: int = SEQUENCE_LENGTH,
                 min_detection_confidence: float = MEDIAPIPE_CONFIDENCE):
        """
        Khởi tạo đối tượng TrichXuatLandmark.

        Tham số:
            sequence_length (int): Số frame cho mỗi chuỗi (sequence)
            min_detection_confidence (float): Ngưỡng tin cậy MediaPipe
        """
        self.sequence_length = sequence_length
        self.hands_detector = mp_hands.Hands(
            static_image_mode=False,       # Chế độ video (track liên tục)
            max_num_hands=MAX_HANDS,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )
        print(f"[INIT] TrichXuatLandmark khởi tạo. Sequence length = {sequence_length} frames")

    def trich_xuat_tu_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Trích xuất vector đặc trưng từ 1 frame ảnh.

        Tham số:
            frame_bgr (np.ndarray): Frame ảnh đọc từ cv2 (BGR format)

        Trả về:
            np.ndarray: Vector shape (126,) - tọa độ 2 tay.
                        Nếu không phát hiện tay, trả về vector 0.
        """
        # Chuyển BGR -> RGB (MediaPipe yêu cầu RGB)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False  # Tối ưu hiệu năng

        # Phát hiện landmarks
        ket_qua = self.hands_detector.process(frame_rgb)

        # Khởi tạo vector đặc trưng rỗng
        vector_2_tay = np.zeros(TOTAL_FEATURES, dtype=np.float32)

        if ket_qua.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(ket_qua.multi_hand_landmarks):
                if i >= MAX_HANDS:
                    break  # Tối đa 2 tay

                # Trích xuất 21 landmarks (x, y, z) của 1 tay
                vi_tri_bat_dau = i * NUM_FEATURES_PER_HAND
                for j, landmark in enumerate(hand_landmarks.landmark):
                    idx = vi_tri_bat_dau + j * 3
                    vector_2_tay[idx]     = landmark.x  # Chuẩn hóa 0-1 theo chiều ngang
                    vector_2_tay[idx + 1] = landmark.y  # Chuẩn hóa 0-1 theo chiều dọc
                    vector_2_tay[idx + 2] = landmark.z  # Độ sâu (tương đối)

        return vector_2_tay

    def xu_ly_video(self, duong_dan_video: str,
                    hien_thi_preview: bool = False) -> np.ndarray | None:
        """
        Đọc toàn bộ video, trích xuất feature sequence.

        Tham số:
            duong_dan_video (str): Đường dẫn đến file video
            hien_thi_preview (bool): Có hiển thị cửa sổ preview không

        Trả về:
            np.ndarray shape (SEQUENCE_LENGTH, 126) hoặc None nếu lỗi
        """
        cap = cv2.VideoCapture(duong_dan_video)

        if not cap.isOpened():
            print(f"  [LỖI] Không mở được video: {duong_dan_video}")
            return None

        frames_features = []  # Lưu feature vector của từng frame
        tong_frame = 0
        frame_co_tay = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            tong_frame += 1
            vector = self.trich_xuat_tu_frame(frame)
            frames_features.append(vector)

            # Kiểm tra frame này có phát hiện tay không
            if np.any(vector != 0):
                frame_co_tay += 1

            # Hiển thị preview nếu yêu cầu (dùng để debug)
            if hien_thi_preview:
                cv2.putText(frame, f"Frame: {tong_frame}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Preview Trich Xuat", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        if hien_thi_preview:
            cv2.destroyAllWindows()

        if tong_frame == 0:
            print(f"  [CẢNH BÁO] Video rỗng: {duong_dan_video}")
            return None

        # Thống kê chất lượng
        ty_le_co_tay = (frame_co_tay / tong_frame) * 100
        if ty_le_co_tay < 30:
            print(f"  [CẢNH BÁO] Tỷ lệ phát hiện tay thấp: {ty_le_co_tay:.1f}% ({duong_dan_video})")

        # ---- PADDING / TRUNCATING về SEQUENCE_LENGTH ----
        sequence = self._chuan_hoa_do_dai(frames_features)
        return sequence

    def _chuan_hoa_do_dai(self, frames_features: list) -> np.ndarray:
        """
        Cắt (truncate) hoặc đệm (pad) chuỗi về đúng SEQUENCE_LENGTH.

        Nếu dài hơn: Lấy SEQUENCE_LENGTH frame ở giữa (trung tâm động tác)
        Nếu ngắn hơn: Đệm 0 ở cuối

        Tham số:
            frames_features (list): Danh sách vector features các frame

        Trả về:
            np.ndarray shape (SEQUENCE_LENGTH, TOTAL_FEATURES)
        """
        n = len(frames_features)
        arr = np.array(frames_features, dtype=np.float32)

        if n >= self.sequence_length:
            # Cắt từ giữa để giữ phần chính của động tác
            bat_dau = (n - self.sequence_length) // 2
            return arr[bat_dau : bat_dau + self.sequence_length]
        else:
            # Đệm 0 vào cuối
            padding = np.zeros(
                (self.sequence_length - n, TOTAL_FEATURES), dtype=np.float32
            )
            return np.vstack([arr, padding])

    def dong(self):
        """Giải phóng tài nguyên MediaPipe."""
        self.hands_detector.close()
        print("[INFO] Đã giải phóng tài nguyên MediaPipe.")


def xu_ly_toan_bo_dataset(video_root: str, output_dir: str,
                           sequence_length: int = SEQUENCE_LENGTH):
    """
    Hàm CHÍNH: Xử lý toàn bộ dataset video, tạo file .npy.

    Tham số:
        video_root (str): Thư mục gốc chứa video (phân loại theo thư mục con)
        output_dir (str): Thư mục xuất file .npy
        sequence_length (int): Độ dài sequence

    Trả về:
        tuple: (X, y) arrays đã được lưu
    """
    print("\n" + "=" * 60)
    print("  BẮT ĐẦU TRÍCH XUẤT ĐẶC TRƯNG TOÀN BỘ DATASET")
    print("=" * 60)

    # --- Quét danh sách video ---
    danh_sach_video = quet_thu_muc_video(video_root)
    if not danh_sach_video:
        print("[LỖI NGHIÊM TRỌNG] Không có video nào để xử lý!")
        return None, None

    # --- Tạo thư mục output ---
    os.makedirs(output_dir, exist_ok=True)

    # --- Khởi tạo bộ trích xuất ---
    trich_xuat = TrichXuatLandmark(sequence_length=sequence_length)

    X_list = []  # Lưu sequences
    y_list = []  # Lưu labels

    tong = len(danh_sach_video)
    loi = 0

    print(f"\n[XỬ LÝ] Bắt đầu xử lý {tong} video...\n")

    for i, (duong_dan, ten_nhan, ma_nhan) in enumerate(danh_sach_video):
        phan_tram = ((i + 1) / tong) * 100
        print(f"  [{i+1:>3}/{tong}] ({phan_tram:>5.1f}%) [{ten_nhan}] {Path(duong_dan).name}")

        if not os.path.exists(duong_dan):
            print(f"    -> [LỖI] File không tồn tại, bỏ qua.")
            loi += 1
            continue

        sequence = trich_xuat.xu_ly_video(duong_dan, hien_thi_preview=False)

        if sequence is None:
            print(f"    -> [LỖI] Không trích xuất được sequence, bỏ qua.")
            loi += 1
            continue

        X_list.append(sequence)
        y_list.append(ma_nhan)
        print(f"    -> [OK] Sequence shape: {sequence.shape}")

    # --- Giải phóng tài nguyên ---
    trich_xuat.dong()

    if not X_list:
        print("\n[LỖI] Không có dữ liệu nào được trích xuất thành công!")
        return None, None

    # --- Chuyển sang NumPy array ---
    X = np.array(X_list, dtype=np.float32)  # (N, SEQ_LEN, 126)
    y = np.array(y_list, dtype=np.int32)    # (N,)

    # Tên các nhãn theo thứ tự index
    ten_nhan_list = [k for k, v in sorted(LABEL_MAP.items(), key=lambda x: x[1])]
    label_names = np.array(ten_nhan_list)

    # --- Lưu ra file .npy ---
    x_path = os.path.join(output_dir, "X_sequences.npy")
    y_path = os.path.join(output_dir, "y_labels.npy")
    label_path = os.path.join(output_dir, "label_names.npy")

    np.save(x_path, X)
    np.save(y_path, y)
    np.save(label_path, label_names)

    # --- In tổng kết ---
    print("\n" + "=" * 60)
    print("  HOÀN TẤT TRÍCH XUẤT ĐẶC TRƯNG")
    print("=" * 60)
    print(f"  Tổng video xử lý  : {tong}")
    print(f"  Thành công         : {len(X_list)}")
    print(f"  Lỗi / Bỏ qua      : {loi}")
    print(f"  Shape X (dữ liệu) : {X.shape}")
    print(f"  Shape y (nhãn)    : {y.shape}")
    print(f"  Số classes        : {len(np.unique(y))}")
    print(f"\n  Đã lưu:")
    print(f"    -> {x_path}")
    print(f"    -> {y_path}")
    print(f"    -> {label_path}")
    print("=" * 60)

    return X, y


def kiem_tra_du_lieu_da_trich_xuat(output_dir: str):
    """
    Kiểm tra và in thông tin về dataset đã trích xuất.

    Tham số:
        output_dir (str): Thư mục chứa các file .npy
    """
    x_path = os.path.join(output_dir, "X_sequences.npy")
    y_path = os.path.join(output_dir, "y_labels.npy")
    label_path = os.path.join(output_dir, "label_names.npy")

    if not all([os.path.exists(p) for p in [x_path, y_path, label_path]]):
        print("[KIỂM TRA] Chưa có dữ liệu đã trích xuất. Hãy chạy trích xuất trước!")
        return

    X = np.load(x_path)
    y = np.load(y_path)
    label_names = np.load(label_path, allow_pickle=True)

    print("\n" + "=" * 50)
    print("KIỂM TRA DATASET ĐÃ TRÍCH XUẤT:")
    print("=" * 50)
    print(f"  Shape X         : {X.shape}")
    print(f"    -> Số mẫu     : {X.shape[0]}")
    print(f"    -> Seq length : {X.shape[1]} frames")
    print(f"    -> Features   : {X.shape[2]} (21 joints x 3 tọa độ x 2 tay)")
    print(f"  Shape y         : {y.shape}")
    print(f"  Các nhãn        : {label_names.tolist()}")
    print(f"  Min X           : {X.min():.4f}")
    print(f"  Max X           : {X.max():.4f}")

    from collections import Counter
    dem = Counter(y.tolist())
    print("\n  Phân phối nhãn:")
    for idx, so in sorted(dem.items()):
        ten = label_names[idx] if idx < len(label_names) else f"class_{idx}"
        print(f"    [{idx}] {ten:<12}: {so} mẫu")
    print("=" * 50)


# ============================================================
#  CHƯƠNG TRÌNH CHÍNH
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Trích xuất Hand Landmarks từ video để huấn luyện mô hình"
    )
    parser.add_argument(
        "--mode", type=str, default="extract",
        choices=["extract", "check"],
        help="'extract': Trích xuất dữ liệu | 'check': Kiểm tra dữ liệu đã trích xuất"
    )
    parser.add_argument(
        "--video_root", type=str, default=VIDEO_ROOT,
        help="Thư mục gốc chứa video (mặc định: part1_thu_thap_du_lieu/raw_videos)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=OUTPUT_DIR,
        help="Thư mục xuất file .npy (mặc định: part2_trich_xuat_dac_trung/dataset)"
    )
    parser.add_argument(
        "--seq_len", type=int, default=SEQUENCE_LENGTH,
        help=f"Độ dài sequence (số frame). Mặc định: {SEQUENCE_LENGTH}"
    )
    args = parser.parse_args()

    if args.mode == "extract":
        X, y = xu_ly_toan_bo_dataset(
            video_root=args.video_root,
            output_dir=args.output_dir,
            sequence_length=args.seq_len
        )
    elif args.mode == "check":
        kiem_tra_du_lieu_da_trich_xuat(args.output_dir)
