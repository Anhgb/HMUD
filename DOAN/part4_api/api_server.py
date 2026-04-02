# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
=============================================================================
MODULE: api_server.py
MÔ TẢ: RESTful API Server bằng FastAPI để nhận diện ngôn ngữ ký hiệu.

ENDPOINTS:
    GET  /              → Kiểm tra server hoạt động (Health Check)
    GET  /info          → Thông tin mô hình (số class, tên class...)
    POST /predict       → Nhận chuỗi landmarks, trả về kết quả nhận diện
    POST /predict/batch → Nhận nhiều sequences, trả về nhiều kết quả

CÁCH CHẠY (trên Ubuntu/VirtualBox hoặc máy local):
    pip install fastapi uvicorn tensorflow numpy
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
    
    Sau đó truy cập: http://localhost:8000/docs  (Swagger UI tự động)

FORMAT REQUEST BODY (POST /predict):
    {
        "sequence": [                  ← list 60 frame
            [0.1, 0.2, ..., 0.9],     ← mỗi frame là list 126 số float
            ...
        ]
    }

FORMAT RESPONSE:
    {
        "ky_hieu": "xin_chao",
        "do_tin_cay": 0.9832,
        "top3": [
            {"nhan": "xin_chao", "xac_suat": 0.9832},
            {"nhan": "tam_biet", "xac_suat": 0.0112},
            {"nhan": "co",       "xac_suat": 0.0056}
        ],
        "thoi_gian_xu_ly_ms": 12.4
    }
=============================================================================
"""

import os
import time
import logging
from typing import List
from contextlib import asynccontextmanager

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# ============================================================
# CẤU HÌNH LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("SLR-API")


# ============================================================
# CẤU HÌNH ĐƯỜNG DẪN
# ============================================================
# Khi chạy trong VirtualBox Ubuntu, chỉnh lại đường dẫn này
MODEL_DIR       = "part3_huan_luyen/models"
MODEL_PATH      = os.path.join(MODEL_DIR, "best_model.h5")
LABEL_PATH      = os.path.join(MODEL_DIR, "label_names.npy")

# Kích thước sequence phải khớp với lúc train
SEQUENCE_LENGTH = 60
NUM_FEATURES    = 126  # 21 landmarks x 3 tọa độ x 2 tay


# ============================================================
# CLASS QUẢN LÝ MÔ HÌNH (Singleton Pattern)
# ============================================================

class QuanLyMoHinh:
    """
    Singleton class tải và quản lý mô hình TensorFlow.
    Tải mô hình 1 lần khi khởi động server, tái sử dụng cho mọi request.
    """

    _model      = None
    _ten_nhan   = None
    _loaded     = False

    @classmethod
    def tai_mo_hinh(cls, model_path: str, label_path: str):
        """
        Tải mô hình và tên nhãn từ đường dẫn.
        Gọi 1 lần khi server khởi động.
        """
        if cls._loaded:
            logger.info("Mô hình đã được tải trước đó, bỏ qua.")
            return

        logger.info(f"Đang tải mô hình từ: {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy file mô hình: {model_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Không tìm thấy file nhãn: {label_path}")

        cls._model    = tf.keras.models.load_model(model_path)
        cls._ten_nhan = np.load(label_path, allow_pickle=True).tolist()
        cls._loaded   = True

        so_class = len(cls._ten_nhan)
        logger.info(f"✅ Tải mô hình thành công! Nhận diện {so_class} ký hiệu.")
        logger.info(f"   Các ký hiệu: {cls._ten_nhan}")

    @classmethod
    def du_doan(cls, sequence: np.ndarray) -> dict:
        """
        Thực hiện dự đoán cho 1 sequence landmarks.

        Tham số:
            sequence (ndarray): shape (60, 126)

        Trả về:
            dict: {'ky_hieu': str, 'do_tin_cay': float, 'top3': list}
        """
        if not cls._loaded:
            raise RuntimeError("Mô hình chưa được tải!")

        # Thêm batch dimension: (1, 60, 126)
        sequence_batch = np.expand_dims(sequence, axis=0).astype(np.float32)

        # Dự đoán
        xac_suat_tat_ca = cls._model.predict(sequence_batch, verbose=0)[0]

        # Index của nhãn có xác suất cao nhất
        idx_du_doan = int(np.argmax(xac_suat_tat_ca))
        do_tin_cay  = float(xac_suat_tat_ca[idx_du_doan])
        ky_hieu     = cls._ten_nhan[idx_du_doan]

        # Top 3 kết quả có xác suất cao nhất
        top3_idx = np.argsort(xac_suat_tat_ca)[::-1][:3]
        top3 = [
            {
                "nhan": cls._ten_nhan[i],
                "xac_suat": round(float(xac_suat_tat_ca[i]), 6)
            }
            for i in top3_idx
        ]

        return {
            "ky_hieu":   ky_hieu,
            "do_tin_cay": round(do_tin_cay, 6),
            "top3":       top3
        }

    @classmethod
    def lay_thong_tin(cls) -> dict:
        """Trả về thông tin mô hình."""
        if not cls._loaded:
            return {"trang_thai": "chưa tải mô hình"}
        return {
            "trang_thai":       "hoạt động",
            "so_class":         len(cls._ten_nhan),
            "ten_cac_ky_hieu":  cls._ten_nhan,
            "sequence_length":  SEQUENCE_LENGTH,
            "num_features":     NUM_FEATURES,
        }


# ============================================================
# SCHEMAS (Pydantic) — Định nghĩa cấu trúc Request/Response
# ============================================================

class SequenceRequest(BaseModel):
    """Schema cho request dự đoán 1 sequence."""

    sequence: List[List[float]] = Field(
        ...,
        description="Chuỗi 60 frame, mỗi frame là 126 số float (tọa độ landmarks)",
        example=[[0.0] * 126] * 60
    )

    @field_validator("sequence")
    @classmethod
    def kiem_tra_sequence(cls, v):
        """Kiểm tra đầu vào hợp lệ."""
        if len(v) != SEQUENCE_LENGTH:
            raise ValueError(
                f"Sequence phải có đúng {SEQUENCE_LENGTH} frame, "
                f"nhưng nhận được {len(v)} frame."
            )
        for i, frame in enumerate(v):
            if len(frame) != NUM_FEATURES:
                raise ValueError(
                    f"Frame {i} phải có {NUM_FEATURES} features, "
                    f"nhưng nhận được {len(frame)}."
                )
        return v


class BatchSequenceRequest(BaseModel):
    """Schema cho request dự đoán nhiều sequences cùng lúc."""

    sequences: List[List[List[float]]] = Field(
        ...,
        description="Danh sách nhiều sequences, mỗi sequence có 60 frame × 126 features"
    )


class PredictResponse(BaseModel):
    """Schema cho kết quả dự đoán."""
    ky_hieu:         str
    do_tin_cay:      float
    top3:            List[dict]
    thoi_gian_xu_ly_ms: float


# ============================================================
# KHỞI TẠO FASTAPI APP
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Tải mô hình khi server khởi động, giải phóng khi tắt."""
    logger.info("🚀 Khởi động Sign Language Recognition API Server...")
    try:
        QuanLyMoHinh.tai_mo_hinh(MODEL_PATH, LABEL_PATH)
    except FileNotFoundError as e:
        logger.warning(f"⚠️  {e}")
        logger.warning("   Server sẽ chạy ở chế độ DEMO (không có mô hình thật).")
    yield
    logger.info("🛑 Đang tắt server...")


app = FastAPI(
    title="Sign Language Recognition API",
    description=(
        "RESTful API nhận diện ngôn ngữ ký hiệu tay cho người câm điếc.\n"
        "Nhận chuỗi Hand Landmarks từ MediaPipe, trả về ký hiệu được nhận diện."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",        # Swagger UI
    redoc_url="/redoc",      # ReDoc UI
)

# ---- Cấu hình CORS (cho phép frontend gọi API từ domain khác) ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Trong production: chỉ định domain cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ENDPOINTS
# ============================================================

@app.get(
    "/",
    summary="Health Check",
    tags=["Hệ thống"]
)
async def health_check():
    """Kiểm tra server có đang hoạt động không."""
    return {
        "trang_thai": "✅ API đang hoạt động",
        "phien_ban":  "1.0.0",
        "mo_hinh":    "Sẵn sàng" if QuanLyMoHinh._loaded else "Chưa tải",
        "mo_ta":      "Sign Language Recognition API - Đồ Án Học Máy Ứng Dụng"
    }


@app.get(
    "/info",
    summary="Thông tin mô hình",
    tags=["Hệ thống"]
)
async def lay_thong_tin_mo_hinh():
    """Trả về thông tin chi tiết về mô hình đang được sử dụng."""
    return QuanLyMoHinh.lay_thong_tin()


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Nhận diện ký hiệu tay",
    tags=["Nhận diện"],
    status_code=status.HTTP_200_OK
)
async def du_doan_ky_hieu(request: SequenceRequest):
    """
    Nhận chuỗi hand landmarks và trả về ký hiệu được nhận diện.

    **Request Body:**
    - `sequence`: Mảng 60 × 126 số float (60 frame, mỗi frame 126 tọa độ MediaPipe)

    **Response:**
    - `ky_hieu`: Tên ký hiệu được nhận diện (ví dụ: "xin_chao")
    - `do_tin_cay`: Độ tin cậy (0.0 → 1.0)
    - `top3`: 3 kết quả có xác suất cao nhất
    - `thoi_gian_xu_ly_ms`: Thời gian xử lý (milliseconds)
    """
    if not QuanLyMoHinh._loaded:
        # Chế độ DEMO: Trả về kết quả giả khi chưa có mô hình
        logger.warning("Yêu cầu dự đoán nhưng mô hình chưa tải — trả về kết quả DEMO")
        return PredictResponse(
            ky_hieu="[DEMO] xin_chao",
            do_tin_cay=0.9999,
            top3=[{"nhan": "[DEMO] xin_chao", "xac_suat": 0.9999}],
            thoi_gian_xu_ly_ms=0.0
        )

    try:
        thoi_diem_bat_dau = time.perf_counter()

        # Chuyển sang NumPy array
        sequence_np = np.array(request.sequence, dtype=np.float32)

        # Thực hiện dự đoán
        ket_qua = QuanLyMoHinh.du_doan(sequence_np)

        # Tính thời gian xử lý
        thoi_gian_ms = (time.perf_counter() - thoi_diem_bat_dau) * 1000

        logger.info(
            f"Dự đoán: '{ket_qua['ky_hieu']}' "
            f"(tin cậy={ket_qua['do_tin_cay']:.2%}, "
            f"time={thoi_gian_ms:.1f}ms)"
        )

        return PredictResponse(
            ky_hieu=ket_qua["ky_hieu"],
            do_tin_cay=ket_qua["do_tin_cay"],
            top3=ket_qua["top3"],
            thoi_gian_xu_ly_ms=round(thoi_gian_ms, 2)
        )

    except Exception as e:
        logger.error(f"Lỗi khi dự đoán: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi xử lý nội bộ: {str(e)}"
        )


@app.post(
    "/predict/batch",
    summary="Nhận diện nhiều sequences cùng lúc",
    tags=["Nhận diện"]
)
async def du_doan_batch(request: BatchSequenceRequest):
    """
    Nhận và xử lý nhiều sequences cùng một lúc.
    Trả về danh sách kết quả tương ứng.
    """
    if not QuanLyMoHinh._loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Mô hình chưa được tải lên server."
        )

    ket_qua_list = []
    for i, seq in enumerate(request.sequences):
        if len(seq) != SEQUENCE_LENGTH or any(len(f) != NUM_FEATURES for f in seq):
            ket_qua_list.append({"index": i, "loi": "Sai kích thước sequence"})
            continue
        try:
            seq_np  = np.array(seq, dtype=np.float32)
            ket_qua = QuanLyMoHinh.du_doan(seq_np)
            ket_qua_list.append({"index": i, **ket_qua})
        except Exception as e:
            ket_qua_list.append({"index": i, "loi": str(e)})

    return {"so_luong": len(ket_qua_list), "ket_qua": ket_qua_list}


# ============================================================
# CHẠY TRỰC TIẾP (dùng khi debug)
# ============================================================

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("  KHỞI ĐỘNG SIGN LANGUAGE RECOGNITION API SERVER")
    print("  Swagger UI: http://localhost:8000/docs")
    print("=" * 60)
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,     # Tự reload khi code thay đổi (chỉ dùng khi dev)
        log_level="info"
    )
