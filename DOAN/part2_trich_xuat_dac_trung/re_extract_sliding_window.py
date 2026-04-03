# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
re_extract_sliding_window.py
Trich xuat lai dataset tu RAW VIDEOS voi ky thuat Sliding Window.
Moi video 4 giay (~120 frame) se tao ra nhieu sequence 60-frame chong nhau.
Ket qua: Tu 30 mau/class -> ~180+ mau/class (tang 6x).
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import sys
from pathlib import Path

# ─── Cấu hình ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DOAN_DIR    = os.path.dirname(BASE_DIR)
VIDEO_ROOT  = os.path.join(DOAN_DIR, "part1_thu_thap_du_lieu", "raw_videos")
OUTPUT_DIR  = os.path.join(DOAN_DIR, "part2_trich_xuat_dac_trung", "dataset")

SEQUENCE_LEN  = 60     # Frame moi sequence - khop voi model
STEP_SIZE     = 10     # Buoc truot: cu 10 frame tao 1 sequence moi (sliding)
MIN_HAND_PCT  = 0.40   # Loai sequence neu < 40% frame co tay
MAX_HANDS     = 2
TOTAL_FEATS   = 63 * MAX_HANDS  # 126

mp_hands = mp.solutions.hands

# ─── Label map từ tên thư mục ─────────────────────────────
sys.path.insert(0, BASE_DIR)
try:
    from parse_label_studio import LABEL_MAP
except Exception:
    LABEL_MAP = {
        "xin_chao": 0, "cam_on": 1, "toi": 2, "ban": 3,
        "yeu": 4, "khong": 5, "co": 6,
        "xin_loi": 8, "tam_biet": 9,
    }

print("="*60)
print("  RE-EXTRACT VOI SLIDING WINDOW")
print("="*60)

# ─── MediaPipe setup ──────────────────────────────────────
detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_HANDS,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_frame(frame_bgr):
    """Trich xuat vector 126 tu 1 frame."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    res = detector.process(rgb)
    vec = np.zeros(TOTAL_FEATS, dtype=np.float32)
    if res.multi_hand_landmarks:
        for i, hl in enumerate(res.multi_hand_landmarks):
            if i >= MAX_HANDS:
                break
            start = i * 63
            for j, lm in enumerate(hl.landmark):
                idx = start + j * 3
                vec[idx], vec[idx+1], vec[idx+2] = lm.x, lm.y, lm.z
    return vec

def read_all_frames(video_path):
    """Doc toan bo frames tu video, tra ve list numpy arrays."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def extract_sliding_sequences(video_path, label_id):
    """
    Trich xuat nhieu sequences tu 1 video bang sliding window.
    Tra ve list of (sequence_array, label_id).
    """
    frames = read_all_frames(video_path)
    if len(frames) < SEQUENCE_LEN:
        print(f"    [WARN] Video qua ngan ({len(frames)} frame), bo qua.")
        return []

    # Trich xuat vector cho tung frame
    all_vecs = [extract_frame(f) for f in frames]
    n = len(all_vecs)

    sequences = []
    for start in range(0, n - SEQUENCE_LEN + 1, STEP_SIZE):
        window = all_vecs[start : start + SEQUENCE_LEN]
        arr = np.array(window, dtype=np.float32)

        # Loc: bo qua sequence it tay qua
        hand_frames = sum(1 for v in window if np.any(v != 0))
        hand_pct = hand_frames / SEQUENCE_LEN
        if hand_pct < MIN_HAND_PCT:
            continue

        sequences.append((arr, label_id))

    return sequences

# ─── Quét tất cả video ────────────────────────────────────
X_list, y_list = [], []
label_names_map = {}  # label_id -> ten nhan

for class_dir in sorted(Path(VIDEO_ROOT).iterdir()):
    if not class_dir.is_dir():
        continue
    ten_nhan = class_dir.name
    if ten_nhan not in LABEL_MAP:
        print(f"[SKIP] '{ten_nhan}' khong co trong LABEL_MAP")
        continue
    ma_nhan = LABEL_MAP[ten_nhan]
    label_names_map[ma_nhan] = ten_nhan

    video_files = list(class_dir.glob("*.avi")) + list(class_dir.glob("*.mp4"))
    class_seqs = []

    print(f"\n[CLASS] {ten_nhan} ({len(video_files)} video)")
    for vf in sorted(video_files):
        seqs = extract_sliding_sequences(str(vf), ma_nhan)
        class_seqs.extend(seqs)
        print(f"    {vf.name}: +{len(seqs)} sequences")

    print(f"  -> Tong {ten_nhan}: {len(class_seqs)} sequences")
    for arr, lbl in class_seqs:
        X_list.append(arr)
        y_list.append(lbl)

detector.close()

if not X_list:
    print("[LOI] Khong trich xuat duoc gi!")
    sys.exit(1)

# ─── Remap label về 0..N-1 liên tục ──────────────────────
y_raw = np.array(y_list, dtype=np.int32)
unique_labels = sorted(np.unique(y_raw).tolist())
label_remap   = {old: new for new, old in enumerate(unique_labels)}
y_remapped    = np.array([label_remap[l] for l in y_raw], dtype=np.int32)
names_arr     = np.array([label_names_map[i] for i in unique_labels])

X_arr = np.array(X_list, dtype=np.float32)  # (N, 60, 126)

# Shuffle
idx = np.random.RandomState(42).permutation(len(X_arr))
X_arr     = X_arr[idx]
y_remapped = y_remapped[idx]

print(f"\n{'='*60}")
print(f"TONG KET:")
print(f"  X shape : {X_arr.shape}")
print(f"  Classes : {names_arr.tolist()}")
from collections import Counter
c = Counter(y_remapped.tolist())
for new_id, cnt in sorted(c.items()):
    print(f"    [{new_id}] {names_arr[new_id]:<12}: {cnt} mau")

# ─── Lưu dataset ──────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.save(os.path.join(OUTPUT_DIR, "X_sequences.npy"), X_arr)
np.save(os.path.join(OUTPUT_DIR, "y_labels.npy"),    y_remapped)
np.save(os.path.join(OUTPUT_DIR, "label_names.npy"), names_arr)

print(f"\nDa luu vao: {OUTPUT_DIR}")
print("Buoc tiep theo: chay retrain_model.py")
