# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

"""
=============================================================================
MODULE: app_desktop.py
MO TA: Giao dien Desktop (CustomTkinter) cho he thong nhan dang
       ngon ngu ky hieu thoi gian thuc qua Webcam. Đã nâng cấp UI Neon & Ghép câu.
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
import pyttsx3
import queue
import re

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
    def __init__(self):
        super().__init__()

        # --- Cấu hình cửa sổ ---
        self.title("🤟 Sign Language AI Assistant — Premium Edition")
        self.geometry("1200x750")
        self.minsize(900, 650)
        self.configure(fg_color="#090c10")  # Màu đen nhám cực ngầu

        # --- Biến trạng thái ---
        self.dang_chay      = False
        self.cap            = None
        self.luong_webcam   = None
        self.buffer_seq     = deque(maxlen=SEQUENCE_LENGTH)
        self.ket_qua_ht     = "---"
        self.tincay_ht      = 0.0
        self.lich_su        = []
        self.tong_frame     = 0
        self.thoi_diem_bd   = None

        # --- Biến ghép câu ---
        self.cau_hien_tai = []
        self.tu_cuoi_cung = None
        self.thoi_gian_nhan_tu_cuoi = time.time()

        # --- Công cụ Text-To-Speech (Nhạc nền) ---
        self.audio_queue = queue.Queue()
        threading.Thread(target=self._xu_ly_am_thanh, daemon=True).start()

        # --- MediaPipe detector ---
        self.hands_detector = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )

        self._lock = threading.Lock()

        # --- Xây dựng UI ---
        self._xay_dung_ui()
        self.protocol("WM_DELETE_WINDOW", self._dong_ung_dung)

    # ========================================================
    # XỬ LÝ ÂM THANH
    # ========================================================
    def _xu_ly_am_thanh(self):
        """Khởi động bộ máy phát âm thanh trong background thread."""
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except:
            pass
        engine = pyttsx3.init()
        engine.setProperty('rate', 140)  # Tốc độ đọc tự nhiên hơn
        while True:
            van_ban = self.audio_queue.get()
            if van_ban is None:
                break
            try:
                engine.say(van_ban)
                engine.runAndWait()
            except Exception as e:
                print(f"[LỖI AUDIO] {e}")

    # ========================================================
    # XÂY DỰNG GIAO DIỆN
    # ========================================================
    def _xay_dung_ui(self):
        # --- HEADER ---
        header = ctk.CTkFrame(self, fg_color="#161b22", corner_radius=0, height=60)
        header.pack(fill="x", padx=0, pady=0)
        header.pack_propagate(False)

        ctk.CTkLabel(
            header,
            text="🤟  SIGN LANGUAGE ASSISTANT  —  Neon UI",
            font=ctk.CTkFont(family="Segoe UI", size=22, weight="bold"),
            text_color="#00ffff"  # Cyan Neon
        ).pack(side="left", padx=20, pady=10)

        self.lbl_trang_thai_header = ctk.CTkLabel(
            header, text="● Chờ khởi động", font=ctk.CTkFont(size=14), text_color="#8b949e"
        )
        self.lbl_trang_thai_header.pack(side="right", padx=20)

        # --- BODY ---
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=15, pady=10)

        col_trai = ctk.CTkFrame(body, fg_color="transparent")
        col_trai.pack(side="left", fill="both", expand=True, padx=(0, 8))

        col_phai = ctk.CTkFrame(body, fg_color="transparent", width=360)
        col_phai.pack(side="right", fill="y", padx=(8, 0))
        col_phai.pack_propagate(False)

        self._tao_panel_webcam(col_trai)
        
        # Băng rôn CÂU chữ khổng lồ dưới Webcam
        self._tao_panel_ghep_cau(col_trai)

        self._tao_panel_ketqua(col_phai)
        self._tao_panel_caidat(col_phai)
        self._tao_panel_lichsu(col_phai)

        # --- FOOTER ---
        self._tao_footer()

    def _tao_panel_webcam(self, parent):
        frame = ctk.CTkFrame(parent, fg_color="#161b22", corner_radius=12)
        frame.pack(fill="both", expand=True, pady=(0, 10))

        self.lbl_video = ctk.CTkLabel(
            frame, text="Nhấn  ▶  Bắt Đầu  để mở camera",
            font=ctk.CTkFont(size=14), text_color="#30363d", fg_color="#0d1117",
            corner_radius=8, width=640, height=480
        )
        self.lbl_video.pack(padx=15, pady=15, expand=True)

        buf_frame = ctk.CTkFrame(frame, fg_color="transparent")
        buf_frame.pack(fill="x", padx=15, pady=(0, 15))

        self.progress_buffer = ctk.CTkProgressBar(
            buf_frame, mode="determinate", progress_color="#ff00ff", fg_color="#21262d", height=8
        )
        self.progress_buffer.set(0)
        self.progress_buffer.pack(side="left", fill="x", expand=True)

    def _tao_panel_ghep_cau(self, parent):
        """Băng rôn lớn hiển thị câu hoàn chỉnh phía dưới camera"""
        frame = ctk.CTkFrame(parent, fg_color="#161b22", corner_radius=12)
        frame.pack(fill="x", side="bottom")

        header_row = ctk.CTkFrame(frame, fg_color="transparent")
        header_row.pack(fill="x", padx=15, pady=(10, 0))

        ctk.CTkLabel(
            header_row, text="💬 Câu hoàn chỉnh:", font=ctk.CTkFont(size=14, weight="bold"), text_color="#8b949e"
        ).pack(side="left")

        btn_row = ctk.CTkFrame(header_row, fg_color="transparent")
        btn_row.pack(side="right")

        ctk.CTkButton(
            btn_row, text="🔊 Đọc Câu", font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#00ffff", hover_color="#00cccc", text_color="#000000", width=90, height=28,
            command=self._doc_toan_bo_cau
        ).pack(side="left", padx=(0, 10))

        ctk.CTkButton(
            btn_row, text="❌ Xóa", font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#da3633", hover_color="#f85149", width=60, height=28,
            command=self._xoa_cau
        ).pack(side="left")

        self.lbl_cau_hien_tai = ctk.CTkLabel(
            frame, text="...", font=ctk.CTkFont(size=22, weight="bold"),
            text_color="#ffffff", wraplength=600, justify="left", anchor="w"
        )
        self.lbl_cau_hien_tai.pack(fill="x", padx=15, pady=(10, 15))

    def _tao_panel_ketqua(self, parent):
        frame = ctk.CTkFrame(parent, fg_color="#161b22", corner_radius=12)
        frame.pack(fill="x", pady=(0, 8))

        ctk.CTkLabel(
            frame, text="🎯  Từ Mới Nhất", font=ctk.CTkFont(size=14, weight="bold"), text_color="#8b949e"
        ).pack(anchor="w", padx=15, pady=(10, 5))

        self.lbl_kyHieu = ctk.CTkLabel(
            frame, text="---", font=ctk.CTkFont(family="Segoe UI Emoji", size=32, weight="bold"), text_color="#00ffff"
        )
        self.lbl_kyHieu.pack(pady=(5, 0))

        self.lbl_tincay = ctk.CTkLabel(
            frame, text="Độ tin cậy: —", font=ctk.CTkFont(size=13), text_color="#8b949e"
        )
        self.lbl_tincay.pack(pady=2)

        self.progress_tincay = ctk.CTkProgressBar(
            frame, mode="determinate", progress_color="#00ffff", fg_color="#21262d", height=12
        )
        self.progress_tincay.set(0)
        self.progress_tincay.pack(fill="x", padx=15, pady=(3, 8))

        self.lbl_top3 = ctk.CTkLabel(
            frame, text="—\n—\n—", font=ctk.CTkFont(size=12), text_color="#8b949e", justify="left"
        )
        self.lbl_top3.pack(anchor="w", padx=20, pady=(2, 10))

    def _tao_panel_caidat(self, parent):
        frame = ctk.CTkFrame(parent, fg_color="#161b22", corner_radius=12)
        frame.pack(fill="x", pady=(0, 8))

        ctk.CTkLabel(
            frame, text="⚙️  Bảng Điều Khiển", font=ctk.CTkFont(size=14, weight="bold"), text_color="#8b949e"
        ).pack(anchor="w", padx=15, pady=(10, 5))

        self.entry_api_url = ctk.CTkEntry(
            frame, placeholder_text=API_URL_DEFAULT, font=ctk.CTkFont(size=12),
            fg_color="#21262d", border_color="#30363d", text_color="#e6edf3"
        )
        self.entry_api_url.insert(0, API_URL_DEFAULT)
        self.entry_api_url.pack(fill="x", padx=15, pady=(2, 8))

        self.slider_nguong = ctk.CTkSlider(
            frame, from_=30, to=95, number_of_steps=13,
            progress_color="#00ffff", button_color="#ffffff", button_hover_color="#00cccc"
        )
        self.slider_nguong.set(60) # Ngưỡng văng câu dễ hơn
        self.slider_nguong.pack(fill="x", padx=15, pady=(0, 15))

        btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
        btn_frame.pack(fill="x", padx=15, pady=(0, 12))

        self.btn_batdau = ctk.CTkButton(
            btn_frame, text="▶  BẮT ĐẦU", font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#238636", hover_color="#2ea043", height=40, command=self._bat_dau
        )
        self.btn_batdau.pack(side="left", fill="x", expand=True, padx=(0, 4))

        self.btn_dung = ctk.CTkButton(
            btn_frame, text="⏹  DỪNG", font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#da3633", hover_color="#f85149", height=40, state="disabled", command=self._dung
        )
        self.btn_dung.pack(side="right", fill="x", expand=True, padx=(4, 0))

    def _tao_panel_lichsu(self, parent):
        frame = ctk.CTkFrame(parent, fg_color="#161b22", corner_radius=12)
        frame.pack(fill="both", expand=True)

        ctk.CTkLabel(
            frame, text="📜  Lịch Sử (Logs)", font=ctk.CTkFont(size=14, weight="bold"), text_color="#8b949e"
        ).pack(anchor="w", padx=15, pady=(10, 5))

        self.txtbox_lichsu = ctk.CTkTextbox(
            frame, font=ctk.CTkFont(family="Consolas", size=11),
            fg_color="#0d1117", text_color="#8b949e", border_width=0, state="disabled"
        )
        self.txtbox_lichsu.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def _tao_footer(self):
        footer = ctk.CTkFrame(self, fg_color="#161b22", corner_radius=0, height=32)
        footer.pack(fill="x", side="bottom")
        footer.pack_propagate(False)

        self.lbl_fps = ctk.CTkLabel(footer, text="FPS: —", font=ctk.CTkFont(size=11), text_color="#6e7681")
        self.lbl_fps.pack(side="left", padx=15)

        self.lbl_so_nhan_dien = ctk.CTkLabel(footer, text="Latency: —", font=ctk.CTkFont(size=11), text_color="#6e7681")
        self.lbl_so_nhan_dien.pack(side="left", padx=15)

    # ========================================================
    # LOGIC GHÉP CÂU + ĐỌC VĂN BẢN
    # ========================================================
    def _xoa_cau(self):
        self.cau_hien_tai.clear()
        self.tu_cuoi_cung = None
        self.lbl_cau_hien_tai.configure(text="...")
        self.audio_queue.put("Đã xóa")

    def _doc_toan_bo_cau(self):
        if not self.cau_hien_tai:
            return
        
        # Lọc sạch text bỏ icon emoji (VD "👋 Xin Chào" -> "Xin Chào")
        doan_mieu_ta = []
        for tu in self.cau_hien_tai:
            chuoi_hien_thi = LABEL_DISPLAY.get(tu, tu)
            chuoi_sach = re.sub(r'[^\w\s]', '', chuoi_hien_thi).strip()
            doan_mieu_ta.append(chuoi_sach)
            
        cau_vong_ngoai = " ".join(doan_mieu_ta)
        self.audio_queue.put(cau_vong_ngoai)

    def _cap_nhat_giao_dien_cau(self):
        if not self.cau_hien_tai:
            self.lbl_cau_hien_tai.configure(text="...")
            return
        cau_text = " ".join([LABEL_DISPLAY.get(tu, tu) for tu in self.cau_hien_tai])
        self.lbl_cau_hien_tai.configure(text=cau_text)

    # ========================================================
    # LOGIC WEBCAM & API
    # ========================================================
    def _bat_dau(self):
        if self.dang_chay: return
        self.cap = cv2.VideoCapture(WEBCAM_INDEX, cv2.CAP_DSHOW) # Dùng DSHOW giúp khởi động camera Windows ổn định hơn
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_H)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():
            self._ghi_lichsu("[LỖI] Không mở được camera.")
            return

        self.dang_chay = True
        self.tong_frame = 0
        self.thoi_diem_bd = time.time()
        self.buffer_seq.clear()

        self.btn_batdau.configure(state="disabled")
        self.btn_dung.configure(state="normal")
        self.lbl_trang_thai_header.configure(text="● Đang Quét (Neon Active)", text_color="#ff00ff")
        
        self.luong_webcam = threading.Thread(target=self._vong_lap_webcam, daemon=True)
        self.luong_webcam.start()

    def _dung(self):
        self.dang_chay = False
        # Chuyển việc giải phóng camera (release) vào cuối thread `_vong_lap_webcam`
        # Tránh lỗi đụng độ bộ nhớ khi đóng camera lúc thread con đang đọc frame
        self.btn_batdau.configure(state="normal")
        self.btn_dung.configure(state="disabled")
        self.lbl_video.configure(image=None, text="Đã dừng AI.")
        self.lbl_trang_thai_header.configure(text="● Đã dừng", text_color="#8b949e")

    def _dong_ung_dung(self):
        self.dang_chay = False
        self.audio_queue.put(None) # Tat TTS
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.destroy()

    def _vong_lap_webcam(self):
        # MÀU NEON CHUẨN CYBERPUNK (BGR format in OpenCV)
        # Các điểm khớp Bàn tay (Cyan bàng bạc)
        landmark_style = mp_draw.DrawingSpec(color=(255, 255, 0), thickness=3, circle_radius=4)
        # Đường nối xương (Hồng rực rỡ)
        connection_style = mp_draw.DrawingSpec(color=(255, 0, 255), thickness=3)

        while self.dang_chay:
            ret, frame = self.cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            self.tong_frame += 1

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            ket_qua_mp = self.hands_detector.process(frame_rgb)
            frame_rgb.flags.writeable = True

            vector = np.zeros(NUM_FEATURES, dtype=np.float32)
            if ket_qua_mp.multi_hand_landmarks:
                for i, hand_lm in enumerate(ket_qua_mp.multi_hand_landmarks):
                    if i >= 2: break
                    bat_dau = i * 63
                    for j, lm in enumerate(hand_lm.landmark):
                        idx = bat_dau + j * 3
                        vector[idx], vector[idx+1], vector[idx+2] = lm.x, lm.y, lm.z

                    # VẼ KHUNG XƯƠNG NEON LÊN FRAME
                    mp_draw.draw_landmarks(
                        frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                        landmark_style, connection_style
                    )

            with self._lock:
                self.buffer_seq.append(vector)
                so_frame_buf = len(self.buffer_seq)

            # Xử lý hiển thị lớp mờ + Text trên frame
            if self.ket_qua_ht != "---" and self.tincay_ht >= (self.slider_nguong.get()/100.0):
                txt = LABEL_DISPLAY.get(self.ket_qua_ht, self.ket_qua_ht)
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, WEBCAM_H - 80), (WEBCAM_W, WEBCAM_H), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                cv2.putText(frame, f"{txt} ({self.tincay_ht*100:.0f}%)", (15, WEBCAM_H - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2, cv2.LINE_AA)

            if so_frame_buf == SEQUENCE_LENGTH:
                with self._lock:
                    seq_copy = list(self.buffer_seq)
                    for _ in range(SEQUENCE_LENGTH // 2):
                        self.buffer_seq.popleft()

                threading.Thread(target=self._goi_api_va_cap_nhat, args=(seq_copy,), daemon=True).start()

            # Render
            fra_rgb_disp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ctk.CTkImage(Image.fromarray(fra_rgb_disp), size=(WEBCAM_W, WEBCAM_H))
            fps = self.tong_frame / (time.time() - self.thoi_diem_bd + 0.001)

            self.after(0, self._cap_nhat_ui_chinh, img, so_frame_buf/SEQUENCE_LENGTH, fps)

        # KHI THOÁT VÒNG LẶP DO BẤM STOP -> GIẢI PHÓNG CAMERA AN TOÀN NHẤT
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def _goi_api_va_cap_nhat(self, sequence):
        url = self.entry_api_url.get().strip()
        nguong = self.slider_nguong.get() / 100.0

        try:
            payload = {"sequence": [v.tolist() for v in sequence]}
            resp = requests.post(f"{url}/predict", json=payload, timeout=3)
            
            if resp.status_code == 200:
                data = resp.json()
                ky_hieu = data.get("ky_hieu", "?")
                tincay  = data.get("do_tin_cay", 0.0)
                top3    = data.get("top3", [])
                ms      = data.get("thoi_gian_xu_ly_ms", 0)

                self.after(0, lambda: self.lbl_so_nhan_dien.configure(text=f"Latency: {ms:.0f}ms"))

                if tincay >= nguong:
                    self.ket_qua_ht = ky_hieu
                    self.tincay_ht  = tincay
                    
                    # LOGIC GHÉP CÂU THÔNG MINH (Debounce)
                    hien_tai = time.time()
                    if self.tu_cuoi_cung != ky_hieu or (hien_tai - self.thoi_gian_nhan_tu_cuoi > 2.0):
                        if ky_hieu != "None":  # Đảm bảo không bắt nhầm frame trống
                            self.cau_hien_tai.append(ky_hieu)
                            self.tu_cuoi_cung = ky_hieu
                            self.thoi_gian_nhan_tu_cuoi = hien_tai
                            
                            # Cú pháp Regex làm mượt text trước khi đọc tiếng Việt
                            hien_thi = LABEL_DISPLAY.get(ky_hieu, ky_hieu)
                            txt_doc = re.sub(r'[^\w\s]', '', hien_thi).strip()
                            self.audio_queue.put(txt_doc)
                            
                            self.after(0, self._cap_nhat_giao_dien_cau)

                    self.after(0, self._cap_nhat_bang_ket_qua, ky_hieu, tincay, top3)

        except requests.exceptions.Timeout: pass
        except requests.exceptions.ConnectionError: pass

    # ========================================================
    # CÁC HÀM TIỆN ÍCH UPDATE GIAO DIỆN KHÁC
    # ========================================================
    def _cap_nhat_ui_chinh(self, img, buff_pct, fps):
        self.lbl_video.configure(image=img, text="")
        self.progress_buffer.set(buff_pct)
        self.lbl_fps.configure(text=f"FPS: {fps:.1f}")

    def _cap_nhat_bang_ket_qua(self, ky_hieu, tincay, top3):
        self.lbl_kyHieu.configure(text=LABEL_DISPLAY.get(ky_hieu, ky_hieu))
        self.lbl_tincay.configure(text=f"Độ tin cậy: {tincay*100:.1f}%")
        self.progress_tincay.set(tincay)

        mau = "#00ffff" if tincay >= 0.8 else ("#ff00ff" if tincay >= 0.6 else "#da3633")
        self.progress_tincay.configure(progress_color=mau)
        self.lbl_kyHieu.configure(text_color=mau)

        t3_txt = "".join([f"› {LABEL_DISPLAY.get(x['nhan'], x['nhan'])}  {x['xac_suat']*100:.1f}%\n" for x in top3[:3]])
        self.lbl_top3.configure(text=t3_txt.strip())
        self._ghi_lichsu(f"• Nhận diện: {ky_hieu} ({tincay*100:.1f}%)")

    def _ghi_lichsu(self, dong):
        self.txtbox_lichsu.configure(state="normal")
        self.txtbox_lichsu.insert("end", dong + "\n")
        self.txtbox_lichsu.see("end")
        self.txtbox_lichsu.configure(state="disabled")

if __name__ == "__main__":
    app = SignLanguageApp()
    app.mainloop()
