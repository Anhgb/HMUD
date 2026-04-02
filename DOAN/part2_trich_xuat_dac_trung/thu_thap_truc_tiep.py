# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
=============================================================================
MODULE: thu_thap_truc_tiep.py
MÔ TẢ: Script thu thập dữ liệu TRỰC TIẾP từ webcam (KHÔNG cần Label Studio).
        Đây là cách nhanh nhất để tạo dataset ngay trên máy tính.
        
CÁCH DÙNG:
    python part2_trich_xuat_dac_trung/thu_thap_truc_tiep.py
    
THAO TÁC:
    - Nhấn [SPACE] để bắt đầu/dừng thu thập
    - Nhấn [Q] để thoát
    - Mỗi lần thu thập = 1 sequence (60 frame)
=============================================================================
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time
from pathlib import Path
from extract_features import TrichXuatLandmark, SEQUENCE_LENGTH, OUTPUT_DIR, LABEL_MAP


# ============================================================
# CẤU HÌNH
# ============================================================
WEBCAM_INDEX = 0          # 0 = webcam mặc định, 1 = webcam ngoài
SO_SEQUENCE_MOI_CLASS = 30  # Số sequence thu thập cho mỗi nhãn
OUTPUT_VIDEO_DIR = "part1_thu_thap_du_lieu/raw_videos"


def ve_landmarks_len_frame(frame, results):
    """Vẽ hand landmarks lên frame để hiển thị."""
    mp_draw = mp.solutions.drawing_utils
    mp_hands_style = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    if results.multi_hand_landmarks:
        for hand_lm in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_lm,
                mp_hands.HAND_CONNECTIONS,
                mp_hands_style.get_default_hand_landmarks_style(),
                mp_hands_style.get_default_hand_connections_style()
            )
    return frame


def thu_thap_webcam():
    """
    Hàm chính: Thu thập dữ liệu từ webcam theo từng nhãn.
    """
    danh_sach_nhan = list(LABEL_MAP.keys())
    print("\n" + "=" * 60)
    print("  THU THẬP DỮ LIỆU TRỰC TIẾP TỪ WEBCAM")
    print("=" * 60)
    print(f"  Các nhãn cần thu thập: {danh_sach_nhan}")
    print(f"  Số sequence/nhãn     : {SO_SEQUENCE_MOI_CLASS}")
    print(f"  Frame/sequence       : {SEQUENCE_LENGTH}")
    print("=" * 60)
    print("\n  >> Nhấn ENTER để bắt đầu...")
    input()

    # Mở webcam
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("[LỖI] Không mở được webcam! Kiểm tra lại thiết bị.")
        return

    trich_xuat = TrichXuatLandmark(sequence_length=SEQUENCE_LENGTH)

    hands_detector = mp.solutions.hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    try:
        # Duyệt qua từng nhãn
        for ten_nhan in danh_sach_nhan:
            ma_nhan = LABEL_MAP[ten_nhan]
            thu_muc_luu = os.path.join(OUTPUT_VIDEO_DIR, ten_nhan)
            os.makedirs(thu_muc_luu, exist_ok=True)

            print(f"\n[NHÃN] Chuẩn bị thu thập: '{ten_nhan}' (mã={ma_nhan})")
            print(f"  -> Thực hiện ký hiệu '{ten_nhan}' trước camera")
            print("  -> Nhấn [SPACE] để bắt đầu thu thập từng sequence")
            print("  -> Nhấn [Q] để bỏ qua nhãn này\n")

            so_sequence_da_thu = 0

            while so_sequence_da_thu < SO_SEQUENCE_MOI_CLASS:
                ret, frame = cap.read()
                if not ret:
                    continue

                # Lật gương để tự nhiên hơn
                frame = cv2.flip(frame, 1)

                # Phát hiện tay để hiển thị
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands_detector.process(frame_rgb)
                frame = ve_landmarks_len_frame(frame, results)

                # Hiển thị thông tin
                trang_thai = f"SAP THU: {ten_nhan.upper()} | {so_sequence_da_thu}/{SO_SEQUENCE_MOI_CLASS}"
                cv2.putText(frame, trang_thai, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, "SPACE=Thu thap | Q=Bo qua nhan", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                cv2.imshow("Thu Thap Du Lieu - Sign Language", frame)
                phim = cv2.waitKey(1) & 0xFF

                if phim == ord('q'):
                    print(f"  -> Bỏ qua nhãn '{ten_nhan}'")
                    break

                if phim == ord(' '):
                    # ---- BẮT ĐẦU THU THẬP 1 SEQUENCE ----
                    print(f"  -> Thu thập sequence {so_sequence_da_thu + 1}/{SO_SEQUENCE_MOI_CLASS}...")

                    # Đếm ngược 3 giây để người dùng chuẩn bị
                    for dem_nguoc in range(3, 0, -1):
                        ret2, f2 = cap.read()
                        f2 = cv2.flip(f2, 1)
                        cv2.putText(f2, f"CHUAN BI: {dem_nguoc}...", (200, 240),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                        cv2.imshow("Thu Thap Du Lieu - Sign Language", f2)
                        cv2.waitKey(1000)

                    # Thu thập SEQUENCE_LENGTH frame
                    frames_features = []
                    ten_file_video = f"{ten_nhan}_{so_sequence_da_thu+1:03d}.avi"
                    duong_dan_luu = os.path.join(thu_muc_luu, ten_file_video)

                    # Cấu hình VideoWriter để lưu video
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    out = cv2.VideoWriter(duong_dan_luu, fourcc, 30.0, (640, 480))

                    for frame_idx in range(SEQUENCE_LENGTH):
                        ret3, frame3 = cap.read()
                        if not ret3:
                            continue
                        frame3 = cv2.flip(frame3, 1)

                        # Lưu video
                        out.write(frame3)

                        # Hiển thị tiến trình
                        phan_tram = int((frame_idx / SEQUENCE_LENGTH) * 100)
                        cv2.rectangle(frame3, (10, 440), (10 + phan_tram * 6, 460),
                                      (0, 255, 0), -1)
                        cv2.putText(frame3, f"DANG GHI: {frame_idx+1}/{SEQUENCE_LENGTH}",
                                    (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.imshow("Thu Thap Du Lieu - Sign Language", frame3)
                        cv2.waitKey(1)

                    out.release()
                    so_sequence_da_thu += 1
                    print(f"     [OK] Đã lưu: {duong_dan_luu}")

            print(f"  [XONG] Nhãn '{ten_nhan}': {so_sequence_da_thu} sequences.")

    finally:
        cap.release()
        hands_detector.close()
        trich_xuat.dong()
        cv2.destroyAllWindows()
        print("\n[HOÀN TẤT] Thu thập dữ liệu xong!")
        print(f"  Video đã lưu tại: {OUTPUT_VIDEO_DIR}")
        print("  Bước tiếp theo: Chạy extract_features.py để trích xuất đặc trưng.")


if __name__ == "__main__":
    thu_thap_webcam()
