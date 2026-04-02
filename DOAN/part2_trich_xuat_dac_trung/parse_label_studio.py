"""
=============================================================================
MODULE: parse_label_studio.py
MÔ TẢ: Đọc và phân tích file JSON được xuất từ Label Studio,
        tạo ra danh sách (đường_dẫn_video, nhãn) để xử lý tiếp.
=============================================================================
"""

import json
import os
from pathlib import Path


# ============================================================
# CẤU HÌNH (thay đổi cho phù hợp với máy của bạn)
# ============================================================
# Đường dẫn đến file annotations.json xuất từ Label Studio
LABEL_STUDIO_JSON_PATH = "annotations.json"

# Thư mục chứa các video thô
VIDEO_ROOT_DIR = "part1_thu_thap_du_lieu/raw_videos"

# Danh sách 10 từ vựng và ánh xạ sang số nguyên (index)
LABEL_MAP = {
    "xin_chao": 0,
    "cam_on":   1,
    "toi":      2,
    "ban":      3,
    "yeu":      4,
    "khong":    5,
    "co":       6,
    "giup_do":  7,
    "xin_loi":  8,
    "tam_biet": 9,
}


def doc_json_label_studio(json_path: str) -> list:
    """
    Đọc file JSON từ Label Studio và trả về danh sách annotations thô.

    Tham số:
        json_path (str): Đường dẫn đến file .json

    Trả về:
        list: Danh sách các task đã gán nhãn
    """
    print(f"[PARSE] Đang đọc file annotations: {json_path}")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"[LỖI] Không tìm thấy file: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"[PARSE] Tìm thấy {len(data)} task trong file JSON.")
    return data


def trich_xuat_duong_dan_va_nhan(task_list: list, video_root: str) -> list:
    """
    Phân tích từng task trong danh sách JSON, lấy ra:
    - Đường dẫn video
    - Nhãn (label) tương ứng

    Tham số:
        task_list (list): Danh sách task từ Label Studio JSON
        video_root (str): Thư mục gốc chứa video

    Trả về:
        list: Danh sách tuple (duong_dan_video, ten_nhan, ma_nhan_so)
    """
    ket_qua = []
    so_bo_qua = 0

    for task in task_list:
        try:
            # --- Lấy đường dẫn video ---
            duong_dan_video = task.get("data", {}).get("video", "")

            # Xử lý trường hợp Label Studio lưu relative path hoặc URL
            # Chuyển về đường dẫn tuyệt đối nếu cần
            if duong_dan_video.startswith("/data/"):
                # Trường hợp Label Studio local storage
                ten_file = Path(duong_dan_video).name
                duong_dan_video = os.path.join(video_root, ten_file)
            elif not os.path.isabs(duong_dan_video):
                duong_dan_video = os.path.join(video_root, duong_dan_video)

            # --- Lấy nhãn từ annotations ---
            annotations = task.get("annotations", [])
            if not annotations:
                print(f"  [CẢNH BÁO] Task ID={task.get('id')} chưa được gán nhãn, bỏ qua.")
                so_bo_qua += 1
                continue

            # Lấy annotation đầu tiên (thường chỉ có 1)
            ket_qua_annotation = annotations[0].get("result", [])
            if not ket_qua_annotation:
                so_bo_qua += 1
                continue

            # Lấy nhãn từ trường "choices"
            ten_nhan = None
            for item in ket_qua_annotation:
                if item.get("type") == "choices":
                    choices = item.get("value", {}).get("choices", [])
                    if choices:
                        ten_nhan = choices[0]  # Lấy nhãn đầu tiên
                        break

            if ten_nhan is None:
                print(f"  [CẢNH BÁO] Không tìm thấy nhãn cho task ID={task.get('id')}, bỏ qua.")
                so_bo_qua += 1
                continue

            # --- Chuyển nhãn sang mã số ---
            ma_nhan = LABEL_MAP.get(ten_nhan, -1)
            if ma_nhan == -1:
                print(f"  [CẢNH BÁO] Nhãn '{ten_nhan}' không có trong LABEL_MAP, bỏ qua.")
                so_bo_qua += 1
                continue

            ket_qua.append((duong_dan_video, ten_nhan, ma_nhan))

        except Exception as e:
            print(f"  [LỖI] Không thể xử lý task ID={task.get('id', '?')}: {e}")
            so_bo_qua += 1
            continue

    print(f"[PARSE] Hoàn tất! Hợp lệ: {len(ket_qua)} | Bỏ qua: {so_bo_qua}")
    return ket_qua


def quet_thu_muc_video(video_root: str) -> list:
    """
    PHƯƠNG ÁN THAY THẾ: Nếu không dùng Label Studio JSON,
    tự động quét thư mục video được tổ chức theo cấu trúc:
    raw_videos/
        xin_chao/  -> nhãn = "xin_chao"
        cam_on/    -> nhãn = "cam_on"
        ...

    Tham số:
        video_root (str): Thư mục gốc chứa các thư mục con theo nhãn

    Trả về:
        list: Danh sách tuple (duong_dan_video, ten_nhan, ma_nhan_so)
    """
    print(f"[QUÉT] Đang quét thư mục: {video_root}")
    ket_qua = []
    dinh_dang_hop_le = {".mp4", ".avi", ".mov", ".mkv"}

    if not os.path.exists(video_root):
        raise FileNotFoundError(f"[LỖI] Không tìm thấy thư mục: {video_root}")

    for ten_nhan in sorted(os.listdir(video_root)):
        thu_muc_nhan = os.path.join(video_root, ten_nhan)
        if not os.path.isdir(thu_muc_nhan):
            continue

        ma_nhan = LABEL_MAP.get(ten_nhan, -1)
        if ma_nhan == -1:
            print(f"  [CẢNH BÁO] Thư mục '{ten_nhan}' không có trong LABEL_MAP, bỏ qua.")
            continue

        so_video = 0
        for ten_file in os.listdir(thu_muc_nhan):
            if Path(ten_file).suffix.lower() in dinh_dang_hop_le:
                duong_dan = os.path.join(thu_muc_nhan, ten_file)
                ket_qua.append((duong_dan, ten_nhan, ma_nhan))
                so_video += 1

        print(f"  -> Nhãn '{ten_nhan}' (mã={ma_nhan}): {so_video} video")

    print(f"[QUÉT] Tổng cộng: {len(ket_qua)} video hợp lệ từ {len(LABEL_MAP)} nhãn.")
    return ket_qua


def in_thong_ke(danh_sach: list):
    """In thống kê phân phối nhãn ra màn hình."""
    from collections import Counter
    dem_nhan = Counter([item[1] for item in danh_sach])

    print("\n" + "=" * 50)
    print("THỐNG KÊ PHÂN PHỐI DỮ LIỆU:")
    print("=" * 50)
    for nhan, so_luong in sorted(dem_nhan.items()):
        thanh = "█" * so_luong
        print(f"  {nhan:<12}: {so_luong:>3} video  {thanh}")
    print("=" * 50 + "\n")


# ============================================================
# CHẠY THỬ MODULE NÀY
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  MODULE PARSE DỮ LIỆU LABEL STUDIO")
    print("=" * 60)

    # --- PHƯƠNG ÁN 1: Đọc từ file JSON Label Studio ---
    if os.path.exists(LABEL_STUDIO_JSON_PATH):
        print("\n[INFO] Sử dụng phương án 1: Đọc từ file JSON Label Studio")
        tasks = doc_json_label_studio(LABEL_STUDIO_JSON_PATH)
        danh_sach_du_lieu = trich_xuat_duong_dan_va_nhan(tasks, VIDEO_ROOT_DIR)

    # --- PHƯƠNG ÁN 2: Quét thư mục (KHUYÊN DÙNG nếu video đã phân loại sẵn) ---
    elif os.path.exists(VIDEO_ROOT_DIR):
        print("\n[INFO] Không tìm thấy JSON, chuyển sang phương án 2: Quét thư mục video")
        danh_sach_du_lieu = quet_thu_muc_video(VIDEO_ROOT_DIR)

    else:
        print("[CẢNH BÁO] Không tìm thấy dữ liệu. Tạo dữ liệu mẫu để test...")
        danh_sach_du_lieu = [
            ("raw_videos/xin_chao/clip_001.mp4", "xin_chao", 0),
            ("raw_videos/cam_on/clip_001.mp4", "cam_on", 1),
        ]

    # In thống kê
    if danh_sach_du_lieu:
        in_thong_ke(danh_sach_du_lieu)
        print(f"[OK] Sẵn sàng xử lý {len(danh_sach_du_lieu)} video.")
    else:
        print("[LỖI] Không có dữ liệu nào được tìm thấy!")
