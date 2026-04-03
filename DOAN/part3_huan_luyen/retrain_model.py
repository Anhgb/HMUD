# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
retrain_model.py - Train lại model với Data Augmentation mạnh để chống overfit
Chạy: python DOAN/part3_huan_luyen/retrain_model.py
"""
import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split

print("="*60)
print("  RETRAIN MODEL VỚI DATA AUGMENTATION")
print("="*60)

# ─── Đường dẫn ───────────────────────────────────────────────
BASE     = os.path.dirname(os.path.abspath(__file__))
DOAN_DIR = os.path.dirname(BASE)

X_PATH     = os.path.join(DOAN_DIR, "part2_trich_xuat_dac_trung/dataset/X_sequences.npy")
Y_PATH     = os.path.join(DOAN_DIR, "part2_trich_xuat_dac_trung/dataset/y_labels.npy")
LABEL_PATH = os.path.join(DOAN_DIR, "part2_trich_xuat_dac_trung/dataset/label_names.npy")
MODEL_DIR  = os.path.join(DOAN_DIR, "part3_huan_luyen/models")
OUT_H5     = os.path.join(MODEL_DIR, "best_model.h5")
OUT_KERAS  = os.path.join(MODEL_DIR, "best_model.keras")

# ─── Tải dữ liệu gốc ─────────────────────────────────────────
X = np.load(X_PATH)
y_raw = np.load(Y_PATH)

if os.path.exists(LABEL_PATH):
    names = np.load(LABEL_PATH, allow_pickle=True)
else:
    names = np.load(os.path.join(MODEL_DIR, "label_names.npy"), allow_pickle=True)

# Remap label ve 0..N-1 lien tuc (tranh truong hop co gap trong)
unique_labels = sorted(np.unique(y_raw).tolist())
label_remap = {old: new for new, old in enumerate(unique_labels)}
y = np.array([label_remap[lbl] for lbl in y_raw], dtype=np.int32)
names_remapped = np.array([names[i] for i in unique_labels])

N_CLASSES = len(unique_labels)
print(f"Du lieu goc: {X.shape}  |  {N_CLASSES} classes: {names_remapped.tolist()}")

# ─── DATA AUGMENTATION ────────────────────────────────────────
def augment_sequence(seq):
    """Tạo 1 biến thể của sequence bằng cách thêm nhiễu + dịch thời gian."""
    aug = seq.copy()
    # 1. Gaussian noise
    noise = np.random.normal(0, 0.008, aug.shape)
    aug = aug + noise
    # 2. Time shift (±5 frames)
    shift = np.random.randint(-5, 6)
    if shift > 0:
        aug = np.vstack([aug[shift:], np.zeros((shift, aug.shape[1]))])
    elif shift < 0:
        aug = np.vstack([np.zeros((-shift, aug.shape[1])), aug[:shift]])
    # 3. Scale nhẹ landmark (±5%)
    scale = np.random.uniform(0.95, 1.05)
    aug = aug * scale
    return aug.astype(np.float32)

# Augment mỗi mẫu thêm 5 lần → 270 * 6 = 1620 mẫu
AUGMENT_FACTOR = 5
X_aug_list = [X]
y_aug_list = [y]
print(f"Đang augment dữ liệu (x{AUGMENT_FACTOR})...")
for _ in range(AUGMENT_FACTOR):
    X_a = np.array([augment_sequence(x) for x in X])
    X_aug_list.append(X_a)
    y_aug_list.append(y)

X_all = np.concatenate(X_aug_list, axis=0)
y_all = np.concatenate(y_aug_list, axis=0)
print(f"Dữ liệu sau augment: {X_all.shape}")

# Shuffle
idx = np.random.permutation(len(X_all))
X_all, y_all = X_all[idx], y_all[idx]

# ─── CHIA DỮ LIỆU ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)
print(f"Train: {X_train.shape}  |  Val: {X_val.shape}  |  Test: {X_test.shape}")

# ─── XÂY DỰNG MODEL MỚI ──────────────────────────────────────
# Dùng Bidirectional LSTM + L2 regularization mạnh hơn
model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True,
                       kernel_regularizer=l2(1e-4)),
                  input_shape=(60, 126)),
    BatchNormalization(),
    Dropout(0.4),

    Bidirectional(LSTM(64, kernel_regularizer=l2(1e-4))),
    BatchNormalization(),
    Dropout(0.4),

    Dense(64, activation='relu', kernel_regularizer=l2(1e-4)),
    Dropout(0.35),

    Dense(N_CLASSES, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ─── CALLBACKS ───────────────────────────────────────────────
os.makedirs(MODEL_DIR, exist_ok=True)
callbacks = [
    ModelCheckpoint(OUT_KERAS, monitor='val_accuracy',
                    save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=20,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=8, min_lr=1e-6, verbose=1)
]

# ─── HUẤN LUYỆN ──────────────────────────────────────────────
print("\n[TRAIN] Bắt đầu huấn luyện...")
history = model.fit(
    X_train, y_train,
    epochs=120,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

# ─── ĐÁNH GIÁ ────────────────────────────────────────────────
print("\n[ĐÁNH GIÁ] Trên tập test:")
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
acc = (y_pred == y_test).mean()
print(f"Test Accuracy: {acc*100:.1f}%")

# Lưu thêm định dạng .h5
model.save(OUT_H5)
print(f"\nModel da luu:\n  {OUT_H5}\n  {OUT_KERAS}")

# Luu label_names da remap (lien tuc tu 0)
np.save(os.path.join(MODEL_DIR, "label_names.npy"), names_remapped)
print(f"Label names: {names_remapped.tolist()}")

# ─── PHÂN PHỐI DỰ ĐOÁN TRÊN INPUT NGẪU NHIÊN (KIỂM TRA BIAS) ─
print("\n[KIEM TRA BIAS] Du doan 10 input ngau nhien:")
np.random.seed(99)
for _ in range(10):
    rnd = np.random.uniform(0.1, 0.9, (1, 60, 126)).astype(np.float32)
    prd = model.predict(rnd, verbose=0)[0]
    idx_top = np.argmax(prd)
    print(f"  {names_remapped[idx_top]:<12} ({prd[idx_top]*100:.1f}%)")

print("\n HOAN TAT! Model moi da duoc luu.")

