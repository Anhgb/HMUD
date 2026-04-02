# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
=============================================================================
MODULE: train_model.py
MÔ TẢ: Xây dựng, huấn luyện và đánh giá mô hình LSTM cho nhận dạng
        ngôn ngữ ký hiệu (Sign Language Recognition).

KIẾN TRÚC MÔ HÌNH:
    Input (60, 126)
        → LSTM(128) → Dropout(0.3)
        → LSTM(64)  → Dropout(0.3)
        → Dense(64, relu) → Dropout(0.2)
        → Dense(10, softmax)    ← 10 class

CÁCH CHẠY:
    python part3_huan_luyen/train_model.py
    python part3_huan_luyen/train_model.py --epochs 100 --batch_size 32
=============================================================================
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")          # Không cần màn hình để lưu ảnh biểu đồ
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, Input
)
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam

# ============================================================
# CẤU HÌNH
# ============================================================
# Đường dẫn dữ liệu (output của extract_features.py)
DATASET_DIR   = "part2_trich_xuat_dac_trung/dataset"
X_PATH        = os.path.join(DATASET_DIR, "X_sequences.npy")
Y_PATH        = os.path.join(DATASET_DIR, "y_labels.npy")
LABEL_PATH    = os.path.join(DATASET_DIR, "label_names.npy")

# Đường dẫn lưu model và kết quả
MODEL_DIR     = "part3_huan_luyen/models"
BIEUDO_DIR    = "part3_huan_luyen/bieu_do"

# Hyperparameters mặc định
DEFAULT_EPOCHS      = 80
DEFAULT_BATCH_SIZE  = 32
DEFAULT_LR          = 1e-3
DEFAULT_VAL_SPLIT   = 0.15    # 15% dành cho validation
DEFAULT_TEST_SPLIT  = 0.15    # 15% dành cho test

# Kiến trúc mô hình
SEQUENCE_LENGTH = 60
NUM_FEATURES    = 126   # 21 landmarks x 3 tọa độ x 2 tay


# ============================================================
# HÀM TẢI VÀ CHUẨN BỊ DỮ LIỆU
# ============================================================

def tai_du_lieu(x_path: str, y_path: str, label_path: str):
    """
    Tải dữ liệu từ file .npy đã được trích xuất.

    Trả về:
        X (ndarray): shape (N, 60, 126)
        y (ndarray): shape (N,) - nhãn số nguyên
        ten_nhan (ndarray): mảng tên các class
    """
    print("[DỮ LIỆU] Đang tải dữ liệu...")

    if not all(os.path.exists(p) for p in [x_path, y_path, label_path]):
        raise FileNotFoundError(
            "[LỖI] Không tìm thấy file .npy!\n"
            "  Hãy chạy extract_features.py trước để tạo dữ liệu."
        )

    X           = np.load(x_path)
    y           = np.load(y_path)
    ten_nhan    = np.load(label_path, allow_pickle=True)

    print(f"  -> X shape    : {X.shape}")
    print(f"  -> y shape    : {y.shape}")
    print(f"  -> Số classes : {len(ten_nhan)} {ten_nhan.tolist()}")

    return X, y, ten_nhan


def chia_du_lieu(X: np.ndarray, y: np.ndarray,
                 val_split: float, test_split: float, random_state: int = 42):
    """
    Chia dữ liệu thành train / validation / test.

    Trả về:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    print(f"\n[CHIA DỮ LIỆU] Val={val_split*100:.0f}% | Test={test_split*100:.0f}%")

    # B1: Tách test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_split, random_state=random_state, stratify=y
    )

    # B2: Tách validation set từ phần còn lại
    val_ratio_adj = val_split / (1.0 - test_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio_adj,
        random_state=random_state,
        stratify=y_temp
    )

    print(f"  -> Train : {len(X_train)} mẫu")
    print(f"  -> Val   : {len(X_val)} mẫu")
    print(f"  -> Test  : {len(X_test)} mẫu")

    return X_train, X_val, X_test, y_train, y_val, y_test


def chuan_hoa_nhan(y_train, y_val, y_test, so_class: int):
    """
    Chuyển nhãn số nguyên sang one-hot encoding cho Keras.

    Trả về:
        y_train_oh, y_val_oh, y_test_oh (one-hot arrays)
    """
    lb = LabelBinarizer()
    lb.fit(range(so_class))

    y_train_oh = lb.transform(y_train)
    y_val_oh   = lb.transform(y_val)
    y_test_oh  = lb.transform(y_test)

    return y_train_oh, y_val_oh, y_test_oh


# ============================================================
# HÀM XÂY DỰNG MÔ HÌNH
# ============================================================

def xay_dung_mo_hinh_lstm(so_class: int,
                           sequence_length: int = SEQUENCE_LENGTH,
                           num_features: int = NUM_FEATURES,
                           learning_rate: float = DEFAULT_LR) -> tf.keras.Model:
    """
    Xây dựng kiến trúc mô hình LSTM cho nhận dạng ký hiệu tay.

    Tham số:
        so_class (int): Số lớp cần phân loại
        sequence_length (int): Độ dài chuỗi (số frame)
        num_features (int): Số features mỗi frame (126)
        learning_rate (float): Tốc độ học

    Trả về:
        model: Mô hình Keras đã được compile
    """
    print("\n[MÔ HÌNH] Đang xây dựng kiến trúc LSTM...")

    model = Sequential(name="SignLanguage_LSTM", layers=[
        # Lớp đầu vào
        Input(shape=(sequence_length, num_features)),

        # ---- KHỐI LSTM 1: Trích xuất đặc trưng cấp cao ----
        LSTM(128, return_sequences=True,
             kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        BatchNormalization(),
        Dropout(0.3),

        # ---- KHỐI LSTM 2: Trích xuất đặc trưng cấp thấp ----
        LSTM(64, return_sequences=False,
             kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        BatchNormalization(),
        Dropout(0.3),

        # ---- KHỐI DENSE: Phân loại ----
        Dense(64, activation="relu",
              kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        Dropout(0.2),

        # ---- OUTPUT LAYER ----
        Dense(so_class, activation="softmax"),
    ])

    # Biên dịch mô hình
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # In kiến trúc
    model.summary()
    print(f"\n[MÔ HÌNH] Số tham số: {model.count_params():,}")

    return model


# ============================================================
# HÀM HUẤN LUYỆN
# ============================================================

def huan_luyen(model: tf.keras.Model,
               X_train, y_train_oh,
               X_val, y_val_oh,
               epochs: int, batch_size: int,
               model_dir: str) -> tf.keras.callbacks.History:
    """
    Huấn luyện mô hình với các callback thông minh.

    Callbacks:
        - ModelCheckpoint: Lưu model tốt nhất theo val_accuracy
        - EarlyStopping: Dừng sớm nếu 20 epoch không cải thiện
        - ReduceLROnPlateau: Giảm learning rate khi bị kẹt

    Trả về:
        history: Lịch sử huấn luyện (loss, accuracy qua các epoch)
    """
    os.makedirs(model_dir, exist_ok=True)
    duong_dan_model = os.path.join(model_dir, "best_model.keras")

    print(f"\n[HUẤN LUYỆN] Bắt đầu! (epochs={epochs}, batch={batch_size})")
    print(f"  -> Model tốt nhất sẽ được lưu tại: {duong_dan_model}\n")

    # ---- Định nghĩa Callbacks ----
    callbacks = [
        ModelCheckpoint(
            filepath=duong_dan_model,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
            mode="max"
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=20,           # Dừng nếu 20 epoch không cải thiện
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,            # Giảm LR xuống còn 50%
            patience=8,
            min_lr=1e-6,
            verbose=1
        ),
    ]

    # ---- Bắt đầu huấn luyện ----
    history = model.fit(
        X_train, y_train_oh,
        validation_data=(X_val, y_val_oh),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    print(f"\n[HOÀN TẤT] Kết thúc huấn luyện sau {len(history.epoch)} epochs.")
    return history


# ============================================================
# HÀM ĐÁNH GIÁ MÔ HÌNH
# ============================================================

def danh_gia_mo_hinh(model: tf.keras.Model,
                     X_test, y_test,
                     ten_nhan: np.ndarray,
                     bieudo_dir: str):
    """
    Đánh giá mô hình trên tập test và lưu các biểu đồ phân tích.

    Xuất ra:
        - Accuracy, F1-score (in ra màn hình)
        - Classification Report chi tiết
        - File ảnh Confusion Matrix
    """
    os.makedirs(bieudo_dir, exist_ok=True)
    so_class = len(ten_nhan)

    print("\n" + "=" * 60)
    print("  ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST")
    print("=" * 60)

    # ---- Dự đoán ----
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred      = np.argmax(y_pred_prob, axis=1)

    # ---- Tính các chỉ số ----
    acc     = accuracy_score(y_test, y_pred)
    f1_mac  = f1_score(y_test, y_pred, average="macro",    zero_division=0)
    f1_wei  = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n  Accuracy (Test Set)   : {acc * 100:.2f}%")
    print(f"  F1-Score (Macro)      : {f1_mac:.4f}")
    print(f"  F1-Score (Weighted)   : {f1_wei:.4f}")

    # ---- Classification Report ----
    print("\n  CLASSIFICATION REPORT:")
    print("  " + "-" * 56)
    report = classification_report(
        y_test, y_pred,
        labels=np.arange(len(ten_nhan)),
        target_names=ten_nhan,
        zero_division=0
    )
    for line in report.split("\n"):
        print("  " + line)

    # ---- Vẽ Confusion Matrix ----
    _ve_confusion_matrix(y_test, y_pred, ten_nhan, bieudo_dir)

    print(f"\n  Biểu đồ đã lưu tại: {bieudo_dir}/")
    print("=" * 60)


def _ve_confusion_matrix(y_test, y_pred, ten_nhan, bieudo_dir):
    """Vẽ và lưu Confusion Matrix dạng heatmap."""
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=ten_nhan,
        yticklabels=ten_nhan,
        linewidths=0.5
    )
    plt.title("Confusion Matrix - Sign Language Recognition", fontsize=16, pad=20)
    plt.ylabel("Nhãn Thực Tế", fontsize=12)
    plt.xlabel("Nhãn Dự Đoán", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    duong_dan = os.path.join(bieudo_dir, "confusion_matrix.png")
    plt.savefig(duong_dan, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  [LƯU] Confusion Matrix: {duong_dan}")


def ve_bieu_do_lich_su(history: tf.keras.callbacks.History, bieudo_dir: str):
    """
    Vẽ 2 biểu đồ Training History:
        1. Accuracy theo epoch (train vs validation)
        2. Loss theo epoch (train vs validation)
    """
    os.makedirs(bieudo_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Kết Quả Huấn Luyện Mô Hình LSTM", fontsize=16, fontweight="bold")

    # ---- Biểu đồ Accuracy ----
    ax1 = axes[0]
    ax1.plot(history.history["accuracy"],     label="Train Accuracy",      color="#2196F3", linewidth=2)
    ax1.plot(history.history["val_accuracy"], label="Validation Accuracy", color="#FF5722", linewidth=2)
    ax1.set_title("Độ Chính Xác (Accuracy)", fontsize=13)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])

    # Đánh dấu điểm tốt nhất
    best_val_acc_epoch = np.argmax(history.history["val_accuracy"])
    best_val_acc       = max(history.history["val_accuracy"])
    ax1.scatter(best_val_acc_epoch, best_val_acc, color="gold", s=100, zorder=5,
                label=f"Best Val: {best_val_acc:.4f} (epoch {best_val_acc_epoch+1})")
    ax1.legend(fontsize=10)

    # ---- Biểu đồ Loss ----
    ax2 = axes[1]
    ax2.plot(history.history["loss"],     label="Train Loss",      color="#4CAF50", linewidth=2)
    ax2.plot(history.history["val_loss"], label="Validation Loss", color="#F44336", linewidth=2)
    ax2.set_title("Giá Trị Hàm Mất Mát (Loss)", fontsize=13)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    duong_dan = os.path.join(bieudo_dir, "training_history.png")
    plt.savefig(duong_dan, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [LƯU] Biểu đồ lịch sử huấn luyện: {duong_dan}")


# ============================================================
# HÀM LƯU MÔ HÌNH VÀ METADATA
# ============================================================

def luu_mo_hinh_va_metadata(model: tf.keras.Model,
                             ten_nhan: np.ndarray,
                             model_dir: str,
                             accuracy: float = 0.0):
    """
    Lưu mô hình dưới 2 format và file metadata.

    Lưu:
        - best_model.keras     (format Keras mới)
        - best_model.h5        (format HDF5 tương thích cũ)
        - label_names.npy      (tên các class)
        - model_info.txt       (thông tin tóm tắt)
    """
    os.makedirs(model_dir, exist_ok=True)

    # Lưu format .keras (khuyên dùng)
    keras_path = os.path.join(model_dir, "best_model.keras")
    model.save(keras_path)
    print(f"  [LƯU] Model (.keras): {keras_path}")

    # Lưu format .h5 (tương thích với FastAPI)
    h5_path = os.path.join(model_dir, "best_model.h5")
    model.save(h5_path)
    print(f"  [LƯU] Model (.h5)   : {h5_path}")

    # Lưu tên nhãn
    label_path = os.path.join(model_dir, "label_names.npy")
    np.save(label_path, ten_nhan)
    print(f"  [LƯU] Tên nhãn     : {label_path}")

    # Lưu thông tin mô hình
    info_path = os.path.join(model_dir, "model_info.txt")
    with open(info_path, "w", encoding="utf-8") as f:
        f.write("=== THÔNG TIN MÔ HÌNH SIGN LANGUAGE RECOGNITION ===\n")
        f.write(f"Kiến trúc       : LSTM (128 → 64) + Dense (64 → {len(ten_nhan)})\n")
        f.write(f"Input Shape     : (60, 126) — 60 frames, 126 features/frame\n")
        f.write(f"Output Classes  : {len(ten_nhan)}\n")
        f.write(f"Tên các Classes : {ten_nhan.tolist()}\n")
        f.write(f"Accuracy (Test) : {accuracy:.4f}\n")
    print(f"  [LƯU] Thông tin    : {info_path}")


# ============================================================
# CHƯƠNG TRÌNH CHÍNH
# ============================================================

def main(epochs: int, batch_size: int, lr: float):
    """Pipeline huấn luyện từ đầu đến cuối."""

    print("\n" + "=" * 60)
    print("  DO AN: NHAN DANG NGON NGU KY HIEU (SLR)")
    print("  PHAN 3: HUAN LUYEN MO HINH LSTM")
    print("=" * 60)

    # ---- B1: Tải dữ liệu ----
    X, y, ten_nhan = tai_du_lieu(X_PATH, Y_PATH, LABEL_PATH)
    so_class = len(ten_nhan)

    # ---- B2: Chia train / val / test ----
    X_train, X_val, X_test, y_train, y_val, y_test = chia_du_lieu(
        X, y, val_split=DEFAULT_VAL_SPLIT, test_split=DEFAULT_TEST_SPLIT
    )

    # ---- B3: One-hot encoding ----
    y_train_oh, y_val_oh, y_test_oh = chuan_hoa_nhan(
        y_train, y_val, y_test, so_class=so_class
    )

    # ---- B4: Xây dựng mô hình ----
    model = xay_dung_mo_hinh_lstm(
        so_class=so_class,
        sequence_length=X.shape[1],
        num_features=X.shape[2],
        learning_rate=lr
    )

    # ---- B5: Huấn luyện ----
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(BIEUDO_DIR, exist_ok=True)

    history = huan_luyen(
        model, X_train, y_train_oh, X_val, y_val_oh,
        epochs=epochs, batch_size=batch_size, model_dir=MODEL_DIR
    )

    # ---- B6: Vẽ biểu đồ lịch sử ----
    ve_bieu_do_lich_su(history, bieudo_dir=BIEUDO_DIR)

    # ---- B7: Đánh giá trên tập Test ----
    danh_gia_mo_hinh(model, X_test, y_test, ten_nhan, bieudo_dir=BIEUDO_DIR)

    # ---- B8: Lưu mô hình ----
    acc = accuracy_score(y_test, np.argmax(model.predict(X_test, verbose=0), axis=1))
    print("\n[LƯU MÔ HÌNH]")
    luu_mo_hinh_va_metadata(model, ten_nhan, MODEL_DIR, accuracy=acc)

    print("\n" + "=" * 60)
    print("  HOAN TAT! MO HINH DA DUOC LUU VAO:")
    print(f"  {MODEL_DIR}/")
    print("  Buoc tiep theo: Chay part4_api/api_server.py")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Huấn luyện mô hình LSTM nhận dạng ngôn ngữ ký hiệu"
    )
    parser.add_argument("--epochs",     type=int,   default=DEFAULT_EPOCHS,     help=f"Số epochs (mặc định: {DEFAULT_EPOCHS})")
    parser.add_argument("--batch_size", type=int,   default=DEFAULT_BATCH_SIZE, help=f"Batch size (mặc định: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--lr",         type=float, default=DEFAULT_LR,         help=f"Learning rate (mặc định: {DEFAULT_LR})")
    args = parser.parse_args()

    main(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
