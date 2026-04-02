"""
=============================================================================
MODULE: app.py (Streamlit Web Application)
MÔ TẢ: Giao diện Web App cho hệ thống nhận dạng ngôn ngữ ký hiệu.

        Luồng hoạt động:
        Webcam → MediaPipe (trích xuất landmarks) → Buffer 60 frame
            → Gửi POST request đến FastAPI → Nhận kết quả JSON
            → Hiển thị text nhận dạng lên video real-time

CÁCH CHẠY:
    streamlit run part5_webapp/app.py

CẤU HÌNH:
    Sửa biến API_URL để trỏ tới FastAPI server (local hoặc VirtualBox)
=============================================================================
"""

import cv2
import time
import threading
import requests
import numpy as np
import mediapipe as mp
import streamlit as st
from collections import deque

# ============================================================
# CẤU HÌNH GIAO DIỆN
# ============================================================
st.set_page_config(
    page_title="🤟 Sign Language Recognition",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CẤU HÌNH KỸ THUẬT
# ============================================================
# URL của FastAPI server
# - Máy local     : http://localhost:8000
# - VirtualBox    : http://192.168.x.x:8000 (IP máy ảo)
API_URL_DEFAULT = "http://localhost:8000"

SEQUENCE_LENGTH = 60    # Phải khớp với lúc train
NUM_FEATURES    = 126   # 21 landmarks x 3 tọa độ x 2 tay
WEBCAM_INDEX    = 0     # 0 = webcam mặc định

# MediaPipe
mp_hands     = mp.solutions.hands
mp_draw      = mp.solutions.drawing_utils
mp_draw_style = mp.solutions.drawing_styles


# ============================================================
# CSS CUSTOM (Giao diện đẹp hơn)
# ============================================================
st.markdown("""
<style>
    /* Nền tối */
    .stApp { background-color: #0e1117; }
    
    /* Header chính */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px 30px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px rgba(102,126,234,0.3);
    }
    .main-header h1 { color: white; font-size: 2.2em; margin: 0; }
    .main-header p  { color: rgba(255,255,255,0.85); margin: 5px 0 0 0; }

    /* Card kết quả */
    .result-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 2px solid;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    .result-card.active   { border-color: #00d2ff; }
    .result-card.inactive { border-color: #444; }

    /* Text kết quả lớn */
    .result-text {
        font-size: 2.5em;
        font-weight: 800;
        background: linear-gradient(90deg, #00d2ff, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }

    /* Badge tin cậy */
    .confidence-badge {
        display: inline-block;
        background: #00d2ff22;
        border: 1px solid #00d2ff;
        color: #00d2ff;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.9em;
        margin-top: 8px;
    }

    /* Buffer indicator */
    .buffer-bar {
        height: 8px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 4px;
        transition: width 0.1s ease;
    }

    /* Status pill */
    .status-online  { color: #00ff88; font-weight: bold; }
    .status-offline { color: #ff4444; font-weight: bold; }
    
    /* Ẩn header Streamlit mặc định */
    #MainMenu, footer { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>🤟 Sign Language Recognition</h1>
    <p>Hệ thống nhận dạng ngôn ngữ ký hiệu thời gian thực · Đồ Án Học Máy Ứng Dụng</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR — CẤU HÌNH
# ============================================================
with st.sidebar:
    st.markdown("## ⚙️ Cấu Hình Hệ Thống")

    api_url = st.text_input(
        "🔗 URL FastAPI Server",
        value=API_URL_DEFAULT,
        help="Địa chỉ FastAPI server. Dùng IP máy ảo nếu chạy trên VirtualBox."
    )

    webcam_idx = st.number_input(
        "📷 Webcam Index",
        min_value=0, max_value=5, value=WEBCAM_INDEX, step=1,
        help="0 = webcam mặc định, 1 = webcam ngoài"
    )

    do_tin_cay_nguong = st.slider(
        "🎯 Ngưỡng độ tin cậy (%)",
        min_value=40, max_value=95, value=70, step=5,
        help="Chỉ hiển thị kết quả khi độ tin cậy >= ngưỡng này"
    ) / 100.0

    st.markdown("---")
    st.markdown("### 🏷️ Từ Vựng Nhận Diện")
    tu_vung = [
        "0: xin_chao 👋", "1: cam_on 🙏", "2: toi 👈",
        "3: ban 👉",       "4: yeu ❤️",   "5: khong 🚫",
        "6: co ✅",        "7: giup_do 🤝","8: xin_loi 😔",
        "9: tam_biet 🖐️"
    ]
    for t in tu_vung:
        st.markdown(f"- `{t}`")

    st.markdown("---")
    # Kiểm tra kết nối API
    if st.button("🔍 Kiểm tra kết nối API"):
        try:
            resp = requests.get(f"{api_url}/info", timeout=5)
            if resp.status_code == 200:
                info = resp.json()
                st.success(f"✅ Kết nối thành công!")
                st.json(info)
            else:
                st.error(f"❌ Server trả về lỗi: {resp.status_code}")
        except Exception as e:
            st.error(f"❌ Không kết nối được: {e}")


# ============================================================
# LAYOUT CHÍNH — 2 CỘT
# ============================================================
col_video, col_ketqua = st.columns([3, 2], gap="large")


# ============================================================
# HÀM XỬ LÝ CORE
# ============================================================

def trich_xuat_landmarks_tu_frame(frame_rgb: np.ndarray, detector) -> np.ndarray:
    """
    Trích xuất vector 126 features từ 1 frame qua MediaPipe.

    Tham số:
        frame_rgb (ndarray): Frame ảnh RGB
        detector: Đối tượng MediaPipe Hands

    Trả về:
        ndarray shape (126,) — zeros nếu không có tay
    """
    frame_rgb.flags.writeable = False
    results = detector.process(frame_rgb)
    frame_rgb.flags.writeable = True

    vector = np.zeros(NUM_FEATURES, dtype=np.float32)

    if results.multi_hand_landmarks:
        for i, hand_lm in enumerate(results.multi_hand_landmarks):
            if i >= 2:
                break
            bat_dau = i * 63  # 21 landmarks x 3 tọa độ
            for j, lm in enumerate(hand_lm.landmark):
                idx = bat_dau + j * 3
                vector[idx],  vector[idx+1], vector[idx+2] = lm.x, lm.y, lm.z

    return vector, results


def gui_request_api(sequence_buffer: list, api_url: str) -> dict | None:
    """
    Gửi sequence landmarks lên FastAPI và nhận kết quả.

    Tham số:
        sequence_buffer (list): Danh sách 60 vectors (60 x 126)
        api_url (str): URL của FastAPI server

    Trả về:
        dict: Kết quả JSON từ server, hoặc None nếu lỗi
    """
    try:
        payload = {"sequence": [v.tolist() for v in sequence_buffer]}
        resp = requests.post(
            f"{api_url}/predict",
            json=payload,
            timeout=5
        )
        if resp.status_code == 200:
            return resp.json()
    except requests.exceptions.ConnectionError:
        return {"loi": "Không kết nối được server"}
    except requests.exceptions.Timeout:
        return {"loi": "Server phản hồi quá chậm"}
    except Exception as e:
        return {"loi": str(e)}
    return None


# ============================================================
# PHẦN ĐIỀU KHIỂN WEBCAM
# ============================================================
with col_video:
    st.markdown("### 📹 Luồng Webcam")

    # Placeholder để cập nhật frame
    video_placeholder = st.empty()
    buffer_placeholder = st.empty()

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        bat_dau = st.button("▶️ Bắt Đầu", type="primary", use_container_width=True)
    with col_btn2:
        dung = st.button("⏹️ Dừng", use_container_width=True)

with col_ketqua:
    st.markdown("### 🎯 Kết Quả Nhận Diện")

    # Placeholders cho kết quả
    ketqua_placeholder  = st.empty()
    tincay_placeholder  = st.empty()
    top3_placeholder    = st.empty()
    stats_placeholder   = st.empty()


# ============================================================
# SESSION STATE — Lưu trạng thái giữa các lần render
# ============================================================
if "dang_chay" not in st.session_state:
    st.session_state.dang_chay = False
if "ket_qua_hien_tai" not in st.session_state:
    st.session_state.ket_qua_hien_tai = "---"
if "do_tin_cay_ht" not in st.session_state:
    st.session_state.do_tin_cay_ht = 0.0
if "lich_su" not in st.session_state:
    st.session_state.lich_su = []


# ============================================================
# HIỂN THỊ KẾT QUẢ MẶC ĐỊNH
# ============================================================
def hien_thi_ket_qua(ky_hieu: str, do_tin_cay: float, top3: list = None):
    """Cập nhật card kết quả nhận diện."""
    mau_border = "#00d2ff" if ky_hieu != "---" else "#444"

    ketqua_placeholder.markdown(f"""
    <div class="result-card" style="border-color: {mau_border};">
        <p style="color: #888; font-size: 0.85em; margin-bottom: 5px;">KÝ HIỆU NHẬN DIỆN</p>
        <p class="result-text">{ky_hieu}</p>
        <div class="confidence-badge">Tin cậy: {do_tin_cay*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

    if top3:
        top3_md = "**🏆 Top 3 kết quả:**\n"
        for i, item in enumerate(top3):
            icon = ["🥇", "🥈", "🥉"][i]
            pct  = item.get("xac_suat", 0) * 100
            top3_md += f"{icon} `{item['nhan']}` — **{pct:.2f}%**\n\n"
        top3_placeholder.markdown(top3_md)


hien_thi_ket_qua("---", 0.0)


# ============================================================
# VÒNG LẶP CHÍNH — NHẬN DẠNG REAL-TIME
# ============================================================
if bat_dau:
    st.session_state.dang_chay = True

if dung:
    st.session_state.dang_chay = False

if st.session_state.dang_chay:
    cap = cv2.VideoCapture(int(webcam_idx))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        st.error("❌ Không thể mở webcam! Kiểm tra lại thiết bị.")
        st.session_state.dang_chay = False
    else:
        # Buffer lưu sequence
        sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)

        # Thống kê
        tong_frame     = 0
        tong_du_doan   = 0
        thoi_gian_bat_dau = time.time()

        # Khởi tạo MediaPipe
        hands = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )

        try:
            while st.session_state.dang_chay:
                ret, frame = cap.read()
                if not ret:
                    break

                # Lật gương
                frame = cv2.flip(frame, 1)
                tong_frame += 1

                # Chuyển BGR → RGB để MediaPipe xử lý
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Trích xuất landmarks
                vector, results = trich_xuat_landmarks_tu_frame(frame_rgb, hands)
                sequence_buffer.append(vector)

                # Vẽ landmarks lên frame
                if results.multi_hand_landmarks:
                    for hand_lm in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(
                            frame, hand_lm,
                            mp_hands.HAND_CONNECTIONS,
                            mp_draw_style.get_default_hand_landmarks_style(),
                            mp_draw_style.get_default_hand_connections_style()
                        )

                # Hiển thị số frame trong buffer
                so_frame_buffer = len(sequence_buffer)
                phan_tram_buffer = int((so_frame_buffer / SEQUENCE_LENGTH) * 100)

                # Vẽ thông tin lên video
                cv2.putText(frame, f"Buffer: {so_frame_buffer}/{SEQUENCE_LENGTH}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Vẽ kết quả hiện tại lên video
                ky_hieu_ht  = st.session_state.ket_qua_hien_tai
                tincay_ht   = st.session_state.do_tin_cay_ht

                if ky_hieu_ht != "---" and tincay_ht >= do_tin_cay_nguong:
                    text_hien_thi = f"{ky_hieu_ht} ({tincay_ht*100:.0f}%)"
                    # Nền text
                    (tw, th), _ = cv2.getTextSize(text_hien_thi, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                    cv2.rectangle(frame, (8, 400), (20 + tw, 450), (0, 0, 0), -1)
                    cv2.putText(frame, text_hien_thi,
                                (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

                # Khi buffer đầy → gửi lên API
                if so_frame_buffer == SEQUENCE_LENGTH:
                    # Gọi API trong luồng riêng để không block webcam
                    seq_copy = list(sequence_buffer)  # Copy để tránh race condition

                    ket_qua_api = gui_request_api(seq_copy, api_url)

                    if ket_qua_api and "ky_hieu" in ket_qua_api:
                        do_tin_cay_nhan = ket_qua_api.get("do_tin_cay", 0)

                        if do_tin_cay_nhan >= do_tin_cay_nguong:
                            st.session_state.ket_qua_hien_tai = ket_qua_api["ky_hieu"]
                            st.session_state.do_tin_cay_ht    = do_tin_cay_nhan
                            st.session_state.lich_su.append(ket_qua_api["ky_hieu"])
                            if len(st.session_state.lich_su) > 10:
                                st.session_state.lich_su = st.session_state.lich_su[-10:]

                            hien_thi_ket_qua(
                                ket_qua_api["ky_hieu"],
                                do_tin_cay_nhan,
                                ket_qua_api.get("top3", [])
                            )

                        tong_du_doan += 1
                    elif ket_qua_api and "loi" in ket_qua_api:
                        stats_placeholder.warning(f"⚠️ Lỗi API: {ket_qua_api['loi']}")

                    # Xóa nửa buffer để tạo cửa sổ trượt (sliding window)
                    for _ in range(SEQUENCE_LENGTH // 2):
                        if sequence_buffer:
                            sequence_buffer.popleft()

                # Hiển thị frame lên Streamlit
                frame_rgb_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb_display, channels="RGB", use_container_width=True)

                # Buffer progress bar
                buffer_placeholder.markdown(f"""
                    <p style="color:#888; font-size:0.8em; margin-bottom:4px;">
                        Buffer sequence: {so_frame_buffer}/{SEQUENCE_LENGTH} frames
                    </p>
                    <div style="background:#222; border-radius:4px; height:8px;">
                        <div class="buffer-bar" style="width:{phan_tram_buffer}%;"></div>
                    </div>
                """, unsafe_allow_html=True)

                # Thống kê
                thoi_gian_chay = time.time() - thoi_gian_bat_dau
                fps = tong_frame / thoi_gian_chay if thoi_gian_chay > 0 else 0
                stats_placeholder.markdown(f"""
                    📊 **Thống kê:** FPS ≈ {fps:.1f} | 
                    Tổng frame: {tong_frame} | 
                    Số lần nhận diện: {tong_du_doan}
                """)

        finally:
            cap.release()
            hands.close()
            st.session_state.dang_chay = False
            st.rerun()

else:
    # Hiển thị placeholder khi chưa bắt đầu
    video_placeholder.markdown("""
    <div style="background:#1a1a2e; border:2px dashed #444; border-radius:12px;
                height:380px; display:flex; align-items:center; justify-content:center;
                text-align:center; color:#666;">
        <div>
            <div style="font-size:4em;">📷</div>
            <p style="font-size:1.2em; margin-top:10px;">Nhấn <b style="color:#00d2ff">▶️ Bắt Đầu</b> để mở webcam</p>
            <p style="font-size:0.85em;">MediaPipe sẽ tự động phát hiện và theo dõi tay bạn</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# LỊCH SỬ NHẬN DIỆN
# ============================================================
if st.session_state.lich_su:
    st.markdown("---")
    st.markdown("### 📜 Lịch Sử Nhận Diện Gần Đây")
    st.write(" → ".join(f"`{k}`" for k in st.session_state.lich_su[-10:]))

    if st.button("🗑️ Xóa lịch sử"):
        st.session_state.lich_su = []
        st.session_state.ket_qua_hien_tai = "---"
        st.session_state.do_tin_cay_ht    = 0.0
        st.rerun()


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#555; font-size:0.8em; padding:10px 0">
    🤟 Sign Language Recognition System · 
    Đồ Án Môn Học Máy Ứng Dụng · 
    Stack: Python · MediaPipe · LSTM · FastAPI · Streamlit
</div>
""", unsafe_allow_html=True)
