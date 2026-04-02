# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

"""
=============================================================================
MODULE: app_desktop.py
MO TA: Giao dien Desktop (CustomTkinter) cho he thong nhan dang
       ngon ngu ky hieu thoi gian thuc qua Webcam.

LUONG HOAT DONG:
    Webcam -> OpenCV -> MediaPipe (126 features/frame)
    -> Buffer 60 frame -> POST /predict (FastAPI)
    -> Hien thi ky hieu tren video + panel ket qua

CACH CHAY:
    python part5_webapp/app_desktop.py
=============================================================================
"""

import cv2
import time
import threading
import requests
import numpy as np
import mediapipe as mp
import customtkinter as ctk
from PIL import Image, ImageTk
from collections import deque


# ============================================================
# CAU HINH GIAO DIEN
# ============================================================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ============================================================
# CAU HINH KY THUAT
# ============================================================
API_URL_DEFAULT     = "http://localhost:8000"
SEQUENCE_LENGTH     = 60          # So frame moi sequence (phai khop luc train)
NUM_FEATURES        = 126         # 21 landmarks x 3 toa do x 2 tay
WEBCAM_W, WEBCAM_H  = 640, 480
WEBCAM_INDEX        = 0

# MediaPipe
mp_hands      = mp.solutions.hands
mp_draw       = mp.solutions.drawing_utils
mp_draw_style = mp.solutions.drawing_styles

# Ten day du ky hieu (hien thi dep hon)
LABEL_DISPLAY = {
    "xin_chao":  "👋 Xin Chào",
    "cam_on":    "🙏 Cảm Ơn",
    "toi":       "👈 Tôi",
    "ban":       "👉 Bạn",
    "yeu":       "❤️ Yêu",
    "khong":     "🚫 Không",
    "co":        "✅ Có",
    "giup_do":   "🤝 Giúp Đỡ",
    "xin_loi":   "😔 Xin Lỗi",
    "tam_biet":  "🖐️ Tạm Biệt",
}


# ============================================================
# CLASS CHÍNH — GIAO DIỆN DESKTOP
# ============================================================

class SignLanguageApp(ctk.CTk):
    """
    Lớp giao diện Desktop chính cho hệ thống nhận dạng ngôn ngữ ký hiệu.
    Kế thừa từ customtkinter.CTk.
    """

    def __init__(self):
        super().__init__()

        # --- Cấu hình cửa sổ ---
        self.title("🤟 Sign Language Recognition — Desktop App")
        self.geometry("1200x720")
        self.minsize(900, 600)
        self.configure(fg_color="#0d1117")

        # --- Biến trạng thái ---
        self.dang_chay      = False          # Có đang phát webcam không
        self.cap            = None           # VideoCapture
        self.luong_webcam   = None           # Thread webcam
        self.buffer_seq     = deque(maxlen=SEQUENCE_LENGTH)  # Buffer sequence
        self.ket_qua_ht     = "---"          # Kết quả hiện tại
        self.tincay_ht      = 0.0
        self.lich_su        = []             # Lịch sử nhận diện
        self.tong_frame     = 0
        self.thoi_diem_bd   = None

        # --- MediaPipe detector ---
        self.hands_detector = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )

        # Khóa thread để tránh race condition
        self._lock = threading.Lock()

        # --- Xây dựng UI ---
        self._xay_dung_ui()

        # Xử lý đóng cửa sổ
        self.protocol("WM_DELETE_WINDOW", self._dong_ung_dung)

    # ========================================================
    # XÂY DỰNG GIAO DIỆN
    # ========================================================

    def _xay_dung_ui(self):
        """Tạo toàn bộ layout giao diện."""

        # --- HEADER ---
        header = ctk.CTkFrame(self, fg_color="#161b22", corner_radius=0, height=60)
        header.pack(fill="x", padx=0, pady=0)
        header.pack_propagate(False)

        ctk.CTkLabel(
            header,
            text="🤟  SIGN LANGUAGE RECOGNITION  —  Real-time AI",
            font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold"),
            text_color="#58a6ff"
        ).pack(side="left", padx=20, pady=10)

        self.lbl_trang_thai_header = ctk.CTkLabel(
            header,
            text="● Chờ khởi động",
            font=ctk.CTkFont(size=13),
            text_color="#8b949e"
        )
        self.lbl_trang_thai_header.pack(side="right", padx=20)

        # --- BODY (2 cột) ---
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=15, pady=10)

        # Cột trái: Webcam
        col_trai = ctk.CTkFrame(body, fg_color="transparent")
        col_trai.pack(side="left", fill="both", expand=True, padx=(0, 8))

        # Cột phải: Kết quả + Cài đặt
        col_phai = ctk.CTkFrame(body, fg_color="transparent", width=340)
        col_phai.pack(side="right", fill="y", padx=(8, 0))
        col_phai.pack_propagate(False)

        self._tao_panel_webcam(col_trai)
        self._tao_panel_ketqua(col_phai)
        self._tao_panel_caidat(col_phai)
        self._tao_panel_lichsu(col_phai)

        # --- FOOTER ---
        self._tao_footer()

    def _tao_panel_webcam(self, parent):
        """Tạo panel hiển thị webcam."""
        frame = ctk.CTkFrame(parent, fg_color="#161b22", corner_radius=12)
        frame.pack(fill="both", expand=True)

        # Tiêu đề panel
        ctk.CTkLabel(
            frame,
            text="📹  Luồng Camera",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#8b949e"
        ).pack(anchor="w", padx=15, pady=(10, 5))

        # Khung hiển thị video
        self.lbl_video = ctk.CTkLabel(
            frame,
            text="Nhấn  ▶  Bắt Đầu  để mở camera",
            font=ctk.CTkFont(size=14),
            text_color="#30363d",
            fg_color="#0d1117",
            corner_radius=8,
            width=640, height=480
        )
        self.lbl_video.pack(padx=15, pady=5)

        # Thanh buffer
        buf_frame = ctk.CTkFrame(frame, fg_color="transparent")
        buf_frame.pack(fill="x", padx=15, pady=(0, 5))

        ctk.CTkLabel(
            buf_frame, text="Buffer:", font=ctk.CTkFont(size=11),
            text_color="#8b949e"
        ).pack(side="left", padx=(0, 8))

        self.progress_buffer = ctk.CTkProgressBar(
            buf_frame, mode="determinate",
            progress_color="#238636", fg_color="#21262d", height=10
        )
        self.progress_buffer.set(0)
        self.progress_buffer.pack(side="left", fill="x", expand=True)

        self.lbl_buffer_val = ctk.CTkLabel(
            buf_frame, text="0/60", font=ctk.CTkFont(size=11),
            text_color="#8b949e", width=45
        )
        self.lbl_buffer_val.pack(side="left", padx=(8, 0))

    def _tao_panel_ketqua(self, parent):
        """Tạo panel hiển thị kết quả nhận diện."""
        frame = ctk.CTkFrame(parent, fg_color="#161b22", corner_radius=12)
        frame.pack(fill="x", pady=(0, 8))

        ctk.CTkLabel(
            frame,
            text="🎯  Kết Quả Nhận Diện",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#8b949e"
        ).pack(anchor="w", padx=15, pady=(10, 5))

        # Ký hiệu chính
        self.lbl_kyHieu = ctk.CTkLabel(
            frame,
            text="---",
            font=ctk.CTkFont(family="Segoe UI Emoji", size=36, weight="bold"),
            text_color="#58a6ff"
        )
        self.lbl_kyHieu.pack(pady=(5, 0))

        # Độ tin cậy
        self.lbl_tincay = ctk.CTkLabel(
            frame,
            text="Độ tin cậy: —",
            font=ctk.CTkFont(size=13),
            text_color="#8b949e"
        )
        self.lbl_tincay.pack(pady=2)

        # Thanh tin cậy
        self.progress_tincay = ctk.CTkProgressBar(
            frame, mode="determinate",
            progress_color="#1f6feb", fg_color="#21262d", height=12
        )
        self.progress_tincay.set(0)
        self.progress_tincay.pack(fill="x", padx=15, pady=(3, 8))

        # Top 3
        ctk.CTkLabel(
            frame, text="Top 3:",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#6e7681"
        ).pack(anchor="w", padx=15)

        self.lbl_top3 = ctk.CTkLabel(
            frame,
            text="—\n—\n—",
            font=ctk.CTkFont(size=12),
            text_color="#8b949e",
            justify="left"
        )
        self.lbl_top3.pack(anchor="w", padx=20, pady=(2, 10))

    def _tao_panel_caidat(self, parent):
        """Tạo panel cài đặt và điều khiển."""
        frame = ctk.CTkFrame(parent, fg_color="#161b22", corner_radius=12)
        frame.pack(fill="x", pady=(0, 8))

        ctk.CTkLabel(
            frame,
            text="⚙️  Cài Đặt",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#8b949e"
        ).pack(anchor="w", padx=15, pady=(10, 5))

        # URL API
        ctk.CTkLabel(
            frame, text="FastAPI Server URL:",
            font=ctk.CTkFont(size=12), text_color="#8b949e"
        ).pack(anchor="w", padx=15)

        self.entry_api_url = ctk.CTkEntry(
            frame,
            placeholder_text=API_URL_DEFAULT,
            font=ctk.CTkFont(size=12),
            fg_color="#21262d", border_color="#30363d",
            text_color="#e6edf3"
        )
        self.entry_api_url.insert(0, API_URL_DEFAULT)
        self.entry_api_url.pack(fill="x", padx=15, pady=(2, 8))

        # Ngưỡng tin cậy
        nguong_frame = ctk.CTkFrame(frame, fg_color="transparent")
        nguong_frame.pack(fill="x", padx=15, pady=(0, 5))

        ctk.CTkLabel(
            nguong_frame, text="Ngưỡng tin cậy:",
            font=ctk.CTkFont(size=12), text_color="#8b949e"
        ).pack(side="left")

        self.lbl_nguong_val = ctk.CTkLabel(
            nguong_frame, text="70%",
            font=ctk.CTkFont(size=12, weight="bold"), text_color="#58a6ff"
        )
        self.lbl_nguong_val.pack(side="right")

        self.slider_nguong = ctk.CTkSlider(
            frame,
            from_=30, to=95, number_of_steps=13,
            command=self._cap_nhat_nguong,
            progress_color="#1f6feb", button_color="#58a6ff",
            button_hover_color="#79c0ff"
        )
        self.slider_nguong.set(70)
        self.slider_nguong.pack(fill="x", padx=15, pady=(0, 5))

        # Webcam index
        cam_frame = ctk.CTkFrame(frame, fg_color="transparent")
        cam_frame.pack(fill="x", padx=15, pady=(0, 8))

        ctk.CTkLabel(
            cam_frame, text="Camera:",
            font=ctk.CTkFont(size=12), text_color="#8b949e"
        ).pack(side="left")

        self.combo_cam = ctk.CTkComboBox(
            cam_frame, values=["0", "1", "2"],
            width=70, font=ctk.CTkFont(size=12),
            fg_color="#21262d", border_color="#30363d",
            button_color="#30363d"
        )
        self.combo_cam.set("0")
        self.combo_cam.pack(side="right")

        # Nút kiểm tra API
        ctk.CTkButton(
            frame,
            text="🔍 Kiểm tra kết nối API",
            font=ctk.CTkFont(size=12),
            fg_color="#21262d", hover_color="#30363d",
            border_width=1, border_color="#30363d",
            command=self._kiem_tra_api
        ).pack(fill="x", padx=15, pady=(0, 10))

        # Nút Bắt đầu / Dừng
        btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
        btn_frame.pack(fill="x", padx=15, pady=(0, 12))

        self.btn_batdau = ctk.CTkButton(
            btn_frame,
            text="▶  Bắt Đầu",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#238636", hover_color="#2ea043",
            height=40, command=self._bat_dau
        )
        self.btn_batdau.pack(side="left", fill="x", expand=True, padx=(0, 4))

        self.btn_dung = ctk.CTkButton(
            btn_frame,
            text="⏹  Dừng",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#da3633", hover_color="#f85149",
            height=40, state="disabled",
            command=self._dung
        )
        self.btn_dung.pack(side="right", fill="x", expand=True, padx=(4, 0))

    def _tao_panel_lichsu(self, parent):
        """Tạo panel lịch sử nhận diện."""
        frame = ctk.CTkFrame(parent, fg_color="#161b22", corner_radius=12)
        frame.pack(fill="both", expand=True)

        header_row = ctk.CTkFrame(frame, fg_color="transparent")
        header_row.pack(fill="x", padx=15, pady=(10, 5))

        ctk.CTkLabel(
            header_row,
            text="📜  Lịch Sử Nhận Diện",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#8b949e"
        ).pack(side="left")

        ctk.CTkButton(
            header_row,
            text="Xóa",
            font=ctk.CTkFont(size=11),
            fg_color="transparent", hover_color="#21262d",
            text_color="#6e7681", width=40, height=24,
            command=self._xoa_lich_su
        ).pack(side="right")

        self.txtbox_lichsu = ctk.CTkTextbox(
            frame,
            font=ctk.CTkFont(family="Consolas", size=12),
            fg_color="#0d1117", text_color="#8b949e",
            border_width=0, state="disabled"
        )
        self.txtbox_lichsu.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def _tao_footer(self):
        """Tạo thanh thống kê dưới cùng."""
        footer = ctk.CTkFrame(self, fg_color="#161b22", corner_radius=0, height=32)
        footer.pack(fill="x", side="bottom")
        footer.pack_propagate(False)

        self.lbl_fps = ctk.CTkLabel(
            footer, text="FPS: —",
            font=ctk.CTkFont(size=11), text_color="#6e7681"
        )
        self.lbl_fps.pack(side="left", padx=15)

        self.lbl_so_nhan_dien = ctk.CTkLabel(
            footer, text="Nhận diện: 0 lần",
            font=ctk.CTkFont(size=11), text_color="#6e7681"
        )
        self.lbl_so_nhan_dien.pack(side="left", padx=15)

        ctk.CTkLabel(
            footer,
            text="MediaPipe  ·  LSTM  ·  FastAPI  ·  CustomTkinter",
            font=ctk.CTkFont(size=10), text_color="#30363d"
        ).pack(side="right", padx=15)

    # ========================================================
    # LOGIC ĐIỀU KHIỂN
    # ========================================================

    def _cap_nhat_nguong(self, val):
        """Cập nhật label khi kéo slider ngưỡng."""
        self.lbl_nguong_val.configure(text=f"{int(val)}%")

    def _kiem_tra_api(self):
        """Kiểm tra kết nối tới FastAPI server."""
        url = self.entry_api_url.get().strip()
        self._ghi_lichsu(f"[CHÚ Ý] Đang kiểm tra: {url}/info ...")

        def _check():
            try:
                resp = requests.get(f"{url}/info", timeout=5)
                if resp.status_code == 200:
                    info = resp.json()
                    so_class = info.get("so_class", "?")
                    trang_thai = info.get("trang_thai", "?")
                    self.after(0, self._ghi_lichsu,
                               f"[OK] Kết nối thành công! Trạng thái: {trang_thai} | {so_class} lớp")
                else:
                    self.after(0, self._ghi_lichsu,
                               f"[LỖI] Server trả về: HTTP {resp.status_code}")
            except requests.exceptions.ConnectionError:
                self.after(0, self._ghi_lichsu,
                           "[LỖI] Không kết nối được server. Kiểm tra URL + FastAPI đang chạy?")
            except Exception as e:
                self.after(0, self._ghi_lichsu, f"[LỖI] {e}")

        threading.Thread(target=_check, daemon=True).start()

    def _bat_dau(self):
        """Khởi động luồng webcam."""
        if self.dang_chay:
            return

        cam_idx = int(self.combo_cam.get())
        self.cap = cv2.VideoCapture(cam_idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WEBCAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_H)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():
            self._ghi_lichsu("[LỖI] Không mở được camera! Kiểm tra lại thiết bị.")
            return

        self.dang_chay     = True
        self.tong_frame    = 0
        self.thoi_diem_bd  = time.time()
        self.buffer_seq.clear()

        # Cập nhật UI
        self.btn_batdau.configure(state="disabled")
        self.btn_dung.configure(state="normal")
        self.lbl_trang_thai_header.configure(
            text="● Đang nhận diện", text_color="#3fb950"
        )
        self._ghi_lichsu("▶ Bắt đầu thu thập landmarks...")

        # Khởi động thread webcam
        self.luong_webcam = threading.Thread(
            target=self._vong_lap_webcam, daemon=True
        )
        self.luong_webcam.start()

    def _dung(self):
        """Dừng webcam và đặt lại trạng thái."""
        self.dang_chay = False

        if self.cap and self.cap.isOpened():
            self.cap.release()

        # Cập nhật UI
        self.btn_batdau.configure(state="normal")
        self.btn_dung.configure(state="disabled")
        self.lbl_trang_thai_header.configure(
            text="● Đã dừng", text_color="#8b949e"
        )
        self.lbl_video.configure(image=None, text="Camera đã dừng. Nhấn ▶ để tiếp tục.")
        self.progress_buffer.set(0)
        self.lbl_buffer_val.configure(text="0/60")
        self._ghi_lichsu("⏹ Đã dừng.")

    def _xoa_lich_su(self):
        """Xóa toàn bộ lịch sử nhận diện."""
        self.lich_su.clear()
        self.txtbox_lichsu.configure(state="normal")
        self.txtbox_lichsu.delete("1.0", "end")
        self.txtbox_lichsu.configure(state="disabled")

    def _dong_ung_dung(self):
        """Xử lý đóng cửa sổ an toàn."""
        self.dang_chay = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.hands_detector.close()
        self.destroy()

    # ========================================================
    # VÒNG LẶP WEBCAM (CHẠY TRONG THREAD RIÊNG)
    # ========================================================

    def _vong_lap_webcam(self):
        """
        Vòng lặp chính: đọc frame → trích xuất landmarks
        → buffer → gửi API → cập nhật UI.
        Chạy trong thread daemon riêng.
        """
        so_lan_nhan_dien = 0

        while self.dang_chay:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Lật gương
            frame = cv2.flip(frame, 1)
            self.tong_frame += 1

            # --- Trích xuất landmarks ---
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            ket_qua_mp = self.hands_detector.process(frame_rgb)
            frame_rgb.flags.writeable = True

            vector = self._trich_xuat_vector(ket_qua_mp)

            with self._lock:
                self.buffer_seq.append(vector)
                so_frame_buf = len(self.buffer_seq)

            # --- Vẽ landmarks lên frame ---
            if ket_qua_mp.multi_hand_landmarks:
                for hand_lm in ket_qua_mp.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame, hand_lm,
                        mp_hands.HAND_CONNECTIONS,
                        mp_draw_style.get_default_hand_landmarks_style(),
                        mp_draw_style.get_default_hand_connections_style()
                    )

            # --- Vẽ kết quả lên frame ---
            self._ve_ket_qua_len_frame(frame)

            # --- Khi buffer đầy: gửi API ---
            if so_frame_buf == SEQUENCE_LENGTH:
                with self._lock:
                    seq_copy = list(self.buffer_seq)
                    # Sliding window: xóa nửa buffer
                    for _ in range(SEQUENCE_LENGTH // 2):
                        if self.buffer_seq:
                            self.buffer_seq.popleft()

                # Gọi API trong thread riêng để không block webcam
                threading.Thread(
                    target=self._goi_api_va_cap_nhat,
                    args=(seq_copy,),
                    daemon=True
                ).start()
                so_lan_nhan_dien += 1

            # --- Cập nhật UI (phải dùng after() vì Tkinter không thread-safe) ---
            frame_rgb_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb_display)
            ctk_img = ctk.CTkImage(
                light_image=pil_img,
                dark_image=pil_img,
                size=(WEBCAM_W, WEBCAM_H)
            )
            phan_tram_buf = so_frame_buf / SEQUENCE_LENGTH

            # Tính FPS
            elapsed = time.time() - self.thoi_diem_bd
            fps = self.tong_frame / elapsed if elapsed > 0 else 0

            self.after(0, self._cap_nhat_ui_webcam,
                       ctk_img, phan_tram_buf, so_frame_buf,
                       fps, so_lan_nhan_dien)

        # Thread kết thúc
        print("[WEBCAM] Vòng lặp kết thúc.")

    def _trich_xuat_vector(self, results) -> np.ndarray:
        """
        Trích xuất vector 126 features từ kết quả MediaPipe.

        Trả về:
            ndarray shape (126,) — zeros nếu không có tay
        """
        vector = np.zeros(NUM_FEATURES, dtype=np.float32)
        if results.multi_hand_landmarks:
            for i, hand_lm in enumerate(results.multi_hand_landmarks):
                if i >= 2:
                    break
                bat_dau = i * 63
                for j, lm in enumerate(hand_lm.landmark):
                    idx = bat_dau + j * 3
                    vector[idx], vector[idx+1], vector[idx+2] = lm.x, lm.y, lm.z
        return vector

    def _ve_ket_qua_len_frame(self, frame: np.ndarray):
        """Vẽ ký hiệu đang nhận diện đè lên frame OpenCV."""
        ky_hieu = self.ket_qua_ht
        tincay  = self.tincay_ht

        if ky_hieu != "---" and tincay >= (self.slider_nguong.get() / 100.0):
            text = LABEL_DISPLAY.get(ky_hieu, ky_hieu)
            # Nền mờ
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, WEBCAM_H - 70), (WEBCAM_W, WEBCAM_H), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            # Text kết quả
            cv2.putText(
                frame, f"  {ky_hieu}  ({tincay*100:.0f}%)",
                (10, WEBCAM_H - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 100), 2, cv2.LINE_AA
            )

    def _goi_api_va_cap_nhat(self, sequence: list):
        """
        Gửi sequence lên FastAPI và cập nhật kết quả.
        Chạy trong thread riêng.
        """
        url    = self.entry_api_url.get().strip()
        nguong = self.slider_nguong.get() / 100.0

        try:
            payload = {"sequence": [v.tolist() for v in sequence]}
            resp = requests.post(f"{url}/predict", json=payload, timeout=5)

            if resp.status_code == 200:
                data = resp.json()
                ky_hieu  = data.get("ky_hieu", "?")
                tincay   = data.get("do_tin_cay", 0.0)
                top3     = data.get("top3", [])
                ms       = data.get("thoi_gian_xu_ly_ms", 0)

                if tincay >= nguong:
                    self.ket_qua_ht = ky_hieu
                    self.tincay_ht  = tincay
                    self.lich_su.append(ky_hieu)
                    # Cập nhật UI trên main thread
                    self.after(0, self._cap_nhat_ui_ketqua,
                               ky_hieu, tincay, top3, ms)

            elif resp.status_code != 200:
                self.after(0, self._ghi_lichsu,
                           f"[LỖI API] HTTP {resp.status_code}")

        except requests.exceptions.ConnectionError:
            self.after(0, self._ghi_lichsu,
                       "[LỖI] Mất kết nối API. Kiểm tra FastAPI server!")
        except requests.exceptions.Timeout:
            self.after(0, self._ghi_lichsu, "[CẢNH BÁO] API phản hồi quá chậm.")
        except Exception as e:
            self.after(0, self._ghi_lichsu, f"[LỖI] {e}")

    # ========================================================
    # CẬP NHẬT UI (GỌI TỪ MAIN THREAD QUA after())
    # ========================================================

    def _cap_nhat_ui_webcam(self, ctk_img, phan_tram_buf, so_frame,
                             fps, so_lan_nd):
        """Cập nhật widget video + buffer bar + FPS."""
        self.lbl_video.configure(image=ctk_img, text="")
        self.lbl_video.image = ctk_img  # Giữ reference tránh GC

        self.progress_buffer.set(phan_tram_buf)
        self.lbl_buffer_val.configure(text=f"{so_frame}/{SEQUENCE_LENGTH}")
        self.lbl_fps.configure(text=f"FPS: {fps:.1f}")
        self.lbl_so_nhan_dien.configure(text=f"Nhận diện: {so_lan_nd} lần")

    def _cap_nhat_ui_ketqua(self, ky_hieu, tincay, top3, ms):
        """Cập nhật panel kết quả nhận diện."""
        # Tên đẹp
        ten_dep = LABEL_DISPLAY.get(ky_hieu, ky_hieu)

        self.lbl_kyHieu.configure(text=ten_dep)
        self.lbl_tincay.configure(text=f"Độ tin cậy: {tincay*100:.1f}%")
        self.progress_tincay.set(tincay)

        # Màu theo độ tin cậy
        if tincay >= 0.85:
            mau = "#3fb950"   # Xanh lá
        elif tincay >= 0.65:
            mau = "#f0883e"   # Cam
        else:
            mau = "#da3633"   # Đỏ

        self.progress_tincay.configure(progress_color=mau)
        self.lbl_kyHieu.configure(text_color=mau)

        # Top 3
        top3_text = ""
        icons = ["🥇", "🥈", "🥉"]
        for i, item in enumerate(top3[:3]):
            ten = LABEL_DISPLAY.get(item.get("nhan", ""), item.get("nhan", ""))
            pct = item.get("xac_suat", 0) * 100
            top3_text += f"{icons[i]} {ten}  {pct:.2f}%\n"
        self.lbl_top3.configure(text=top3_text.strip())

        # Ghi lịch sử
        import datetime
        now = datetime.datetime.now().strftime("%H:%M:%S")
        self._ghi_lichsu(
            f"[{now}] {ky_hieu} — {tincay*100:.1f}%  ({ms:.0f}ms)"
        )

    def _ghi_lichsu(self, dong: str):
        """Thêm 1 dòng vào textbox lịch sử."""
        self.txtbox_lichsu.configure(state="normal")
        self.txtbox_lichsu.insert("end", dong + "\n")
        self.txtbox_lichsu.see("end")
        self.txtbox_lichsu.configure(state="disabled")


# ============================================================
# CHẠY ỨNG DỤNG
# ============================================================

if __name__ == "__main__":
    print("Khoi dong Sign Language Recognition Desktop App...")
    app = SignLanguageApp()
    app.mainloop()
