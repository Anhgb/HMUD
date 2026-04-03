# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

"""
=============================================================================
MODULE: app_desktop.py  ·  Version 3.0 — Ultra Premium Edition
MO TA: Giao dien Desktop cho he thong Nhan Dang Ngon Ngu Ky Hieu thoi gian thuc.
       Tinh nang: Neon UI, Ghep Cau, TTS, Thong Ke Phien, Xuat Log, Top-5 Tu.
=============================================================================
"""

import cv2
import time
import datetime
import threading
import requests
import numpy as np
import mediapipe as mp
import customtkinter as ctk
from PIL import Image
from collections import deque, Counter
import pyttsx3
import queue
import re
import os

# ============================================================
# CAU HINH GIAO DIEN
# ============================================================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ============================================================
# CAU HINH KY THUAT
# ============================================================
API_URL_DEFAULT    = "http://localhost:8000"
SEQUENCE_LENGTH    = 60
NUM_FEATURES       = 126
WEBCAM_W, WEBCAM_H = 640, 480
WEBCAM_INDEX       = 0

mp_hands      = mp.solutions.hands
mp_draw       = mp.solutions.drawing_utils

LABEL_DISPLAY = {
    "xin_chao": "👋 Xin Chào",
    "cam_on":   "🙏 Cảm Ơn",
    "toi":      "👈 Tôi",
    "ban":      "👉 Bạn",
    "yeu":      "❤️ Yêu",
    "khong":    "🚫 Không",
    "co":       "✅ Có",
    "giup_do":  "🤝 Giúp Đỡ",
    "xin_loi":  "😔 Xin Lỗi",
    "tam_biet": "🖐️ Tạm Biệt",
}

# Màu sắc chủ đề
C = {
    "bg":       "#090c10",
    "panel":    "#0d1117",
    "border":   "#21262d",
    "text":     "#e6edf3",
    "muted":    "#8b949e",
    "cyan":     "#00ffff",
    "magenta":  "#ff00ff",
    "green":    "#3fb950",
    "orange":   "#f0883e",
    "red":      "#da3633",
    "header":   "#161b22",
}


# ============================================================
# CLASS CHÍNH
# ============================================================
class SignLanguageApp(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("🤟  Sign Language AI  ·  Ultra Premium v3.0")
        self.geometry("1280x800")
        self.minsize(960, 680)
        self.configure(fg_color=C["bg"])

        # --- Biến trạng thái ---
        self.dang_chay   = False
        self.cap         = None
        self.luong_webcam = None
        self.buffer_seq  = deque(maxlen=SEQUENCE_LENGTH)
        self.ket_qua_ht  = "---"
        self.tincay_ht   = 0.0
        self.tong_frame  = 0
        self.thoi_diem_bd = None
        self.co_tay      = False  # Đang phát hiện tay không

        # --- Thống kê phiên ---
        self.so_tu_nhan_dien   = 0
        self.tong_do_tincay    = 0.0
        self.dem_tu            = Counter()  # Tần suất từ

        # --- Ghép câu ---
        self.cau_hien_tai           = []
        self.tu_cuoi_cung           = None
        self.thoi_gian_nhan_tu_cuoi = time.time()

        # --- TTS ---
        self.audio_queue = queue.Queue()
        threading.Thread(target=self._xu_ly_am_thanh, daemon=True).start()

        # --- MediaPipe ---
        self.hands_detector = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )
        self._lock = threading.Lock()

        # --- Build UI ---
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # --- Cập nhật đồng hồ ---
        self._tick_clock()

    # ====================================================
    # TTS
    # ====================================================
    def _xu_ly_am_thanh(self):
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except Exception:
            pass
        engine = pyttsx3.init()
        engine.setProperty('rate', 145)
        while True:
            van_ban = self.audio_queue.get()
            if van_ban is None:
                break
            try:
                engine.say(van_ban)
                engine.runAndWait()
            except Exception as e:
                print(f"[TTS ERROR] {e}")

    # ====================================================
    # ĐỒNG HỒ PHIÊN
    # ====================================================
    def _tick_clock(self):
        if self.dang_chay and self.thoi_diem_bd:
            elapsed = int(time.time() - self.thoi_diem_bd)
            h = elapsed // 3600
            m = (elapsed % 3600) // 60
            s = elapsed % 60
            self.lbl_thoi_gian.configure(text=f"⏱ {h:02d}:{m:02d}:{s:02d}")
        self.after(1000, self._tick_clock)

    # ====================================================
    # XÂY DỰNG UI
    # ====================================================
    def _build_ui(self):
        # ─── HEADER ───────────────────────────────────────
        hdr = ctk.CTkFrame(self, fg_color=C["header"], corner_radius=0, height=56)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        ctk.CTkLabel(
            hdr, text="🤟  SIGN LANGUAGE AI  ·  Neural Interface v3.0",
            font=ctk.CTkFont("Segoe UI", 20, "bold"), text_color=C["cyan"]
        ).pack(side="left", padx=20)

        # Đèn trạng thái bên phải header
        self.lbl_status_hdr = ctk.CTkLabel(
            hdr, text="⬤  Standby", font=ctk.CTkFont(size=13), text_color=C["muted"]
        )
        self.lbl_status_hdr.pack(side="right", padx=20)

        # Chỉ thị phát hiện bàn tay
        self.lbl_hand_detect = ctk.CTkLabel(
            hdr, text="✋ Không có tay", font=ctk.CTkFont(size=12), text_color=C["border"]
        )
        self.lbl_hand_detect.pack(side="right", padx=(0, 30))

        # ─── BODY ─────────────────────────────────────────
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=12, pady=8)

        col_l = ctk.CTkFrame(body, fg_color="transparent")
        col_l.pack(side="left", fill="both", expand=True, padx=(0, 6))

        col_r = ctk.CTkFrame(body, fg_color="transparent", width=380)
        col_r.pack(side="right", fill="y", padx=(6, 0))
        col_r.pack_propagate(False)

        self._build_webcam_panel(col_l)
        self._build_sentence_panel(col_l)  # Băng rôn dưới webcam
        self._build_result_panel(col_r)
        self._build_stats_panel(col_r)
        self._build_controls_panel(col_r)
        self._build_topwords_panel(col_r)

        # ─── FOOTER ───────────────────────────────────────
        self._build_footer()

    # ─── PANEL WEBCAM ─────────────────────────────────────
    def _build_webcam_panel(self, parent):
        frame = ctk.CTkFrame(parent, fg_color=C["panel"], corner_radius=14)
        frame.pack(fill="both", expand=True, pady=(0, 8))

        self.lbl_video = ctk.CTkLabel(
            frame,
            text="Nhấn  ▶  BẮT ĐẦU  để mở Camera",
            font=ctk.CTkFont(size=15), text_color=C["border"],
            fg_color="#0a0e14", corner_radius=10,
            width=640, height=460
        )
        self.lbl_video.pack(padx=12, pady=12, expand=True)

        # Buffer bar
        buf_row = ctk.CTkFrame(frame, fg_color="transparent")
        buf_row.pack(fill="x", padx=12, pady=(0, 10))

        ctk.CTkLabel(buf_row, text="Buffer", font=ctk.CTkFont(size=11),
                     text_color=C["muted"]).pack(side="left", padx=(0, 8))

        self.progress_buf = ctk.CTkProgressBar(
            buf_row, mode="determinate",
            progress_color=C["magenta"], fg_color=C["border"], height=6
        )
        self.progress_buf.set(0)
        self.progress_buf.pack(side="left", fill="x", expand=True)

        self.lbl_buf_val = ctk.CTkLabel(
            buf_row, text="0/60", font=ctk.CTkFont(size=11),
            text_color=C["muted"], width=45
        )
        self.lbl_buf_val.pack(side="left", padx=(8, 0))

    # ─── PANEL GHÉP CÂU ───────────────────────────────────
    def _build_sentence_panel(self, parent):
        frame = ctk.CTkFrame(parent, fg_color=C["panel"], corner_radius=14)
        frame.pack(fill="x", side="bottom")

        top_row = ctk.CTkFrame(frame, fg_color="transparent")
        top_row.pack(fill="x", padx=14, pady=(10, 4))

        ctk.CTkLabel(
            top_row, text="💬  Câu hoàn chỉnh",
            font=ctk.CTkFont(size=13, weight="bold"), text_color=C["muted"]
        ).pack(side="left")

        btn_row = ctk.CTkFrame(top_row, fg_color="transparent")
        btn_row.pack(side="right")

        # Nút đọc câu
        ctk.CTkButton(
            btn_row, text="🔊 Đọc", width=72, height=28,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=C["cyan"], hover_color="#00cccc", text_color="#000",
            command=self._read_sentence
        ).pack(side="left", padx=(0, 6))

        # Nút undo từ cuối
        ctk.CTkButton(
            btn_row, text="↩ Xóa từ cuối", width=110, height=28,
            font=ctk.CTkFont(size=12),
            fg_color=C["border"], hover_color="#30363d", text_color=C["muted"],
            command=self._undo_last_word
        ).pack(side="left", padx=(0, 6))

        # Nút xóa toàn bộ
        ctk.CTkButton(
            btn_row, text="🗑 Xóa hết", width=88, height=28,
            font=ctk.CTkFont(size=12),
            fg_color=C["red"], hover_color="#f85149",
            command=self._clear_sentence
        ).pack(side="left")

        self.lbl_sentence = ctk.CTkLabel(
            frame, text="...",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#ffffff", wraplength=680, justify="left", anchor="w"
        )
        self.lbl_sentence.pack(fill="x", padx=14, pady=(4, 12))

    # ─── PANEL KẾT QUẢ ────────────────────────────────────
    def _build_result_panel(self, parent):
        frame = ctk.CTkFrame(parent, fg_color=C["panel"], corner_radius=14)
        frame.pack(fill="x", pady=(0, 7))

        ctk.CTkLabel(
            frame, text="🎯  Nhận Diện Mới Nhất",
            font=ctk.CTkFont(size=13, weight="bold"), text_color=C["muted"]
        ).pack(anchor="w", padx=14, pady=(10, 4))

        self.lbl_result = ctk.CTkLabel(
            frame, text="— Chưa nhận diện —",
            font=ctk.CTkFont("Segoe UI Emoji", 26, "bold"),
            text_color=C["cyan"]
        )
        self.lbl_result.pack(pady=(0, 2))

        self.lbl_confidence_text = ctk.CTkLabel(
            frame, text="Độ tin cậy : —",
            font=ctk.CTkFont(size=12), text_color=C["muted"]
        )
        self.lbl_confidence_text.pack()

        self.progress_conf = ctk.CTkProgressBar(
            frame, mode="determinate",
            progress_color=C["cyan"], fg_color=C["border"], height=14
        )
        self.progress_conf.set(0)
        self.progress_conf.pack(fill="x", padx=14, pady=(4, 4))

        # Top 3
        ctk.CTkLabel(
            frame, text="Top 3 khả năng:",
            font=ctk.CTkFont(size=11, weight="bold"), text_color=C["border"]
        ).pack(anchor="w", padx=14)

        self.lbl_top3 = ctk.CTkLabel(
            frame, text="—\n—\n—",
            font=ctk.CTkFont("Consolas", 11),
            text_color=C["muted"], justify="left"
        )
        self.lbl_top3.pack(anchor="w", padx=20, pady=(2, 10))

    # ─── PANEL THỐNG KÊ PHIÊN ─────────────────────────────
    def _build_stats_panel(self, parent):
        frame = ctk.CTkFrame(parent, fg_color=C["panel"], corner_radius=14)
        frame.pack(fill="x", pady=(0, 7))

        ctk.CTkLabel(
            frame, text="📊  Thống Kê Phiên",
            font=ctk.CTkFont(size=13, weight="bold"), text_color=C["muted"]
        ).pack(anchor="w", padx=14, pady=(10, 6))

        stats_grid = ctk.CTkFrame(frame, fg_color="transparent")
        stats_grid.pack(fill="x", padx=14, pady=(0, 10))

        def _stat_box(parent, label, init_val, row, col, color):
            box = ctk.CTkFrame(parent, fg_color=C["border"], corner_radius=8)
            box.grid(row=row, column=col, padx=3, pady=3, sticky="ew")
            parent.columnconfigure(col, weight=1)
            ctk.CTkLabel(box, text=label, font=ctk.CTkFont(size=9),
                         text_color=C["muted"]).pack(pady=(5, 0))
            lbl = ctk.CTkLabel(box, text=init_val,
                               font=ctk.CTkFont(size=16, weight="bold"),
                               text_color=color)
            lbl.pack(pady=(0, 5))
            return lbl

        self.lbl_thoi_gian  = _stat_box(stats_grid, "THỜI GIAN", "00:00:00", 0, 0, C["cyan"])
        self.lbl_so_tu      = _stat_box(stats_grid, "SỐ TỪ",    "0",        0, 1, C["green"])
        self.lbl_tb_tc      = _stat_box(stats_grid, "ĐỘ TIN CẬY TB", "—%", 0, 2, C["orange"])

    # ─── PANEL ĐIỀU KHIỂN ─────────────────────────────────
    def _build_controls_panel(self, parent):
        frame = ctk.CTkFrame(parent, fg_color=C["panel"], corner_radius=14)
        frame.pack(fill="x", pady=(0, 7))

        ctk.CTkLabel(
            frame, text="⚙️  Điều Khiển",
            font=ctk.CTkFont(size=13, weight="bold"), text_color=C["muted"]
        ).pack(anchor="w", padx=14, pady=(10, 4))

        # URL API
        self.entry_url = ctk.CTkEntry(
            frame, placeholder_text=API_URL_DEFAULT,
            font=ctk.CTkFont(size=12),
            fg_color=C["border"], border_color="#30363d", text_color=C["text"]
        )
        self.entry_url.insert(0, API_URL_DEFAULT)
        self.entry_url.pack(fill="x", padx=14, pady=(0, 6))

        # Slider ngưỡng + label
        sl_row = ctk.CTkFrame(frame, fg_color="transparent")
        sl_row.pack(fill="x", padx=14)
        ctk.CTkLabel(sl_row, text="Ngưỡng:", font=ctk.CTkFont(size=11),
                     text_color=C["muted"]).pack(side="left")
        self.lbl_threshold = ctk.CTkLabel(
            sl_row, text="60%", font=ctk.CTkFont(size=11, weight="bold"),
            text_color=C["cyan"]
        )
        self.lbl_threshold.pack(side="right")

        self.slider_thresh = ctk.CTkSlider(
            frame, from_=30, to=95, number_of_steps=13,
            command=lambda v: self.lbl_threshold.configure(text=f"{int(v)}%"),
            progress_color=C["cyan"], button_color="#ffffff"
        )
        self.slider_thresh.set(60)
        self.slider_thresh.pack(fill="x", padx=14, pady=(2, 8))

        # Nút Bắt Đầu / Dừng / Xuất log
        btn_row = ctk.CTkFrame(frame, fg_color="transparent")
        btn_row.pack(fill="x", padx=14, pady=(0, 8))

        self.btn_start = ctk.CTkButton(
            btn_row, text="▶  BẮT ĐẦU", height=38,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=C["green"], hover_color="#2ea043",
            command=self._start
        )
        self.btn_start.pack(side="left", fill="x", expand=True, padx=(0, 4))

        self.btn_stop = ctk.CTkButton(
            btn_row, text="⏹  DỪNG", height=38,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=C["red"], hover_color="#f85149",
            state="disabled", command=self._stop
        )
        self.btn_stop.pack(side="right", fill="x", expand=True, padx=(4, 0))

        # Nút xuất log
        ctk.CTkButton(
            frame, text="💾  Xuất Log ra File .txt",
            font=ctk.CTkFont(size=11), height=30,
            fg_color=C["border"], hover_color="#30363d",
            text_color=C["muted"],
            command=self._export_log
        ).pack(fill="x", padx=14, pady=(0, 10))

    # ─── PANEL TOP TỪ ─────────────────────────────────────
    def _build_topwords_panel(self, parent):
        frame = ctk.CTkFrame(parent, fg_color=C["panel"], corner_radius=14)
        frame.pack(fill="both", expand=True)

        top_row = ctk.CTkFrame(frame, fg_color="transparent")
        top_row.pack(fill="x", padx=14, pady=(10, 4))

        ctk.CTkLabel(
            top_row, text="🏆  Từ Nhận Diện Nhiều Nhất",
            font=ctk.CTkFont(size=13, weight="bold"), text_color=C["muted"]
        ).pack(side="left")

        ctk.CTkButton(
            top_row, text="Reset", width=48, height=22,
            font=ctk.CTkFont(size=10), fg_color="transparent",
            hover_color=C["border"], text_color=C["muted"],
            command=self._reset_stats
        ).pack(side="right")

        self.txtbox_log = ctk.CTkTextbox(
            frame, font=ctk.CTkFont("Consolas", 11),
            fg_color="#0a0e14", text_color=C["muted"],
            border_width=0, state="disabled"
        )
        self.txtbox_log.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    # ─── FOOTER ───────────────────────────────────────────
    def _build_footer(self):
        ftr = ctk.CTkFrame(self, fg_color=C["header"], corner_radius=0, height=28)
        ftr.pack(fill="x", side="bottom")
        ftr.pack_propagate(False)

        self.lbl_fps = ctk.CTkLabel(
            ftr, text="FPS: —", font=ctk.CTkFont(size=10), text_color=C["border"]
        )
        self.lbl_fps.pack(side="left", padx=14)

        self.lbl_latency = ctk.CTkLabel(
            ftr, text="Latency: —", font=ctk.CTkFont(size=10), text_color=C["border"]
        )
        self.lbl_latency.pack(side="left", padx=8)

        ctk.CTkLabel(
            ftr, text="MediaPipe · LSTM · FastAPI · TTS · CustomTkinter",
            font=ctk.CTkFont(size=10), text_color=C["border"]
        ).pack(side="right", padx=14)

    # ====================================================
    # LOGIC GHÉP CÂU
    # ====================================================
    def _clear_sentence(self):
        self.cau_hien_tai.clear()
        self.tu_cuoi_cung = None
        self.lbl_sentence.configure(text="...")

    def _undo_last_word(self):
        if self.cau_hien_tai:
            removed = self.cau_hien_tai.pop()
            self._refresh_sentence_label()
            self._log(f"↩ Đã xóa từ: {LABEL_DISPLAY.get(removed, removed)}")

    def _read_sentence(self):
        if not self.cau_hien_tai:
            self.audio_queue.put("Câu rỗng.")
            return
        parts = []
        for tu in self.cau_hien_tai:
            raw = LABEL_DISPLAY.get(tu, tu)
            parts.append(re.sub(r'[^\w\s]', '', raw).strip())
        self.audio_queue.put(" ".join(parts))

    def _refresh_sentence_label(self):
        if not self.cau_hien_tai:
            self.lbl_sentence.configure(text="...")
        else:
            self.lbl_sentence.configure(
                text=" ".join([LABEL_DISPLAY.get(t, t) for t in self.cau_hien_tai])
            )

    # ====================================================
    # XUẤT LOG
    # ====================================================
    def _export_log(self):
        path = os.path.join(
            os.path.dirname(__file__),
            f"lich_su_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        try:
            self.txtbox_log.configure(state="normal")
            content = self.txtbox_log.get("1.0", "end")
            self.txtbox_log.configure(state="disabled")
            with open(path, "w", encoding="utf-8") as f:
                f.write("=== LỊCH SỬ PHIÊN NHẬN DIỆN ===\n")
                f.write(f"Xuất lúc: {datetime.datetime.now()}\n")
                f.write(f"Câu ghép: {' '.join(self.cau_hien_tai)}\n")
                f.write("="*40 + "\n")
                f.write(content)
            self._log(f"💾 Đã xuất log → {os.path.basename(path)}")
        except Exception as e:
            self._log(f"[LỖI] Xuất log thất bại: {e}")

    def _reset_stats(self):
        self.so_tu_nhan_dien = 0
        self.tong_do_tincay  = 0.0
        self.dem_tu.clear()
        self.lbl_so_tu.configure(text="0")
        self.lbl_tb_tc.configure(text="—%")
        self.txtbox_log.configure(state="normal")
        self.txtbox_log.delete("1.0", "end")
        self.txtbox_log.configure(state="disabled")

    # ====================================================
    # WEBCAM
    # ====================================================
    def _start(self):
        if self.dang_chay:
            return
        # Đợi thread cũ kết thúc trước
        if self.luong_webcam and self.luong_webcam.is_alive():
            self.update()
            self.luong_webcam.join(timeout=2.0)

        self.dang_chay     = True
        self.tong_frame    = 0
        self.thoi_diem_bd  = time.time()
        self.buffer_seq.clear()

        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.lbl_status_hdr.configure(text="⬤  Đang chạy", text_color=C["magenta"])

        self.luong_webcam = threading.Thread(target=self._webcam_loop, daemon=True)
        self.luong_webcam.start()

    def _stop(self):
        self.dang_chay = False
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.lbl_trang_thai = self.lbl_status_hdr.configure(
            text="⬤  Đã dừng", text_color=C["muted"]
        )
        self.lbl_video.configure(image=None,
                                 text="Đã dừng  ·  Nhấn ▶ để tiếp tục")
        self.lbl_hand_detect.configure(text="✋ Không có tay", text_color=C["border"])

    def _on_close(self):
        self.dang_chay = False
        self.audio_queue.put(None)
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.destroy()

    # ====================================================
    # VÒNG LẶP WEBCAM (THREAD RIÊNG)
    # ====================================================
    def _webcam_loop(self):
        self.cap = cv2.VideoCapture(WEBCAM_INDEX, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WEBCAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_H)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():
            self.after(0, lambda: self._log("[LỖI] Không mở được Camera!"))
            self.after(0, self._stop)
            return

        # Neon style
        lm_style = mp_draw.DrawingSpec(color=(255, 255, 0), thickness=3, circle_radius=4)
        cn_style = mp_draw.DrawingSpec(color=(255, 0, 255), thickness=2)

        while self.dang_chay:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            self.tong_frame += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            mp_res = self.hands_detector.process(rgb)
            rgb.flags.writeable = True

            # Trích xuất vector đặc trưng
            vector = np.zeros(NUM_FEATURES, dtype=np.float32)
            co_tay_frame = False
            if mp_res.multi_hand_landmarks:
                co_tay_frame = True
                for i, hand_lm in enumerate(mp_res.multi_hand_landmarks):
                    if i >= 2:
                        break
                    start = i * 63
                    for j, lm in enumerate(hand_lm.landmark):
                        idx = start + j * 3
                        vector[idx], vector[idx+1], vector[idx+2] = lm.x, lm.y, lm.z
                    mp_draw.draw_landmarks(frame, hand_lm,
                                          mp_hands.HAND_CONNECTIONS,
                                          lm_style, cn_style)

            # Cập nhật chỉ thị tay (gửi về UI thread)
            self.after(0, self._update_hand_indicator, co_tay_frame)

            with self._lock:
                self.buffer_seq.append(vector)
                buf_len = len(self.buffer_seq)

            # Overlay chữ kết quả lên frame
            if self.ket_qua_ht != "---" and self.tincay_ht >= (self.slider_thresh.get() / 100):
                ov = frame.copy()
                cv2.rectangle(ov, (0, WEBCAM_H - 80), (WEBCAM_W, WEBCAM_H),
                              (0, 0, 0), -1)
                cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
                cv2.putText(
                    frame,
                    f"{LABEL_DISPLAY.get(self.ket_qua_ht, self.ket_qua_ht)}  "
                    f"({self.tincay_ht * 100:.0f}%)",
                    (14, WEBCAM_H - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 0), 2, cv2.LINE_AA
                )

            # Khi buffer đầy → gọi API
            if buf_len == SEQUENCE_LENGTH:
                with self._lock:
                    seq_copy = list(self.buffer_seq)
                    for _ in range(SEQUENCE_LENGTH // 2):
                        self.buffer_seq.popleft()
                threading.Thread(
                    target=self._call_api, args=(seq_copy,), daemon=True
                ).start()

            # Cập nhật UI
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ctk_img = ctk.CTkImage(Image.fromarray(img_rgb), size=(WEBCAM_W, WEBCAM_H))
            fps = self.tong_frame / (time.time() - self.thoi_diem_bd + 0.001)
            self.after(0, self._ui_update_video, ctk_img, buf_len / SEQUENCE_LENGTH, fps)

        if self.cap and self.cap.isOpened():
            self.cap.release()

    # ====================================================
    # GỌI API
    # ====================================================
    def _call_api(self, sequence):
        url    = self.entry_url.get().strip()
        nguong = self.slider_thresh.get() / 100.0
        try:
            payload = {"sequence": [v.tolist() for v in sequence]}
            resp = requests.post(f"{url}/predict", json=payload, timeout=3)
            if resp.status_code == 200:
                data    = resp.json()
                ky_hieu = data.get("ky_hieu", "?")
                tincay  = data.get("do_tin_cay", 0.0)
                top3    = data.get("top3", [])
                ms      = data.get("thoi_gian_xu_ly_ms", 0)

                self.after(0, self.lbl_latency.configure,
                           {"text": f"Latency: {ms:.0f}ms"})

                if tincay >= nguong and ky_hieu not in ("None", "?"):
                    self.ket_qua_ht = ky_hieu
                    self.tincay_ht  = tincay

                    # Ghép câu (debounce 2 giây)
                    now = time.time()
                    if (self.tu_cuoi_cung != ky_hieu or
                            now - self.thoi_gian_nhan_tu_cuoi > 2.0):
                        self.cau_hien_tai.append(ky_hieu)
                        self.tu_cuoi_cung = ky_hieu
                        self.thoi_gian_nhan_tu_cuoi = now

                        # TTS tự động đọc từng từ
                        raw = LABEL_DISPLAY.get(ky_hieu, ky_hieu)
                        self.audio_queue.put(re.sub(r'[^\w\s]', '', raw).strip())

                        # Cập nhật thống kê
                        self.so_tu_nhan_dien += 1
                        self.tong_do_tincay  += tincay
                        self.dem_tu[ky_hieu] += 1

                        self.after(0, self._ui_update_stats)
                        self.after(0, self._refresh_sentence_label)

                    self.after(0, self._ui_update_result, ky_hieu, tincay, top3)

        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError):
            pass
        except Exception as e:
            self.after(0, self._log, f"[LỖI API] {e}")

    # ====================================================
    # CẬP NHẬT UI
    # ====================================================
    def _update_hand_indicator(self, co_tay: bool):
        if co_tay:
            self.lbl_hand_detect.configure(
                text="✋ ĐANG PHÁT HIỆN TAY", text_color=C["cyan"]
            )
        else:
            self.lbl_hand_detect.configure(
                text="✋ Không có tay", text_color=C["border"]
            )

    def _ui_update_video(self, img, buf_pct, fps):
        self.lbl_video.configure(image=img, text="")
        self.progress_buf.set(buf_pct)
        self.lbl_buf_val.configure(text=f"{int(buf_pct * SEQUENCE_LENGTH)}/{SEQUENCE_LENGTH}")
        self.lbl_fps.configure(text=f"FPS: {fps:.1f}")

    def _ui_update_result(self, ky_hieu, tincay, top3):
        ten = LABEL_DISPLAY.get(ky_hieu, ky_hieu)
        self.lbl_result.configure(text=ten)
        self.lbl_confidence_text.configure(text=f"Độ tin cậy : {tincay*100:.1f}%")
        self.progress_conf.set(tincay)

        # Màu gradient theo confidence
        if tincay >= 0.85:
            mau = C["green"]
        elif tincay >= 0.65:
            mau = C["cyan"]
        elif tincay >= 0.50:
            mau = C["orange"]
        else:
            mau = C["red"]
        self.progress_conf.configure(progress_color=mau)
        self.lbl_result.configure(text_color=mau)

        # Top 3
        icons = ["🥇", "🥈", "🥉"]
        lines = []
        for i, item in enumerate(top3[:3]):
            n = LABEL_DISPLAY.get(item.get("nhan", ""), item.get("nhan", ""))
            p = item.get("xac_suat", 0) * 100
            lines.append(f"{icons[i]} {n:<20} {p:.1f}%")
        self.lbl_top3.configure(text="\n".join(lines) if lines else "—\n—\n—")

        self._log(f"• {ky_hieu} — {tincay*100:.1f}%")

    def _ui_update_stats(self):
        self.lbl_so_tu.configure(text=str(self.so_tu_nhan_dien))
        if self.so_tu_nhan_dien > 0:
            tb = (self.tong_do_tincay / self.so_tu_nhan_dien) * 100
            self.lbl_tb_tc.configure(text=f"{tb:.0f}%")

        # Top từ
        top5 = self.dem_tu.most_common(5)
        if top5:
            self.txtbox_log.configure(state="normal")
            self.txtbox_log.delete("1.0", "end")
            self.txtbox_log.insert("end", "─── Top Từ ───\n")
            for tu, count in top5:
                bar = "█" * count
                self.txtbox_log.insert(
                    "end", f"  {LABEL_DISPLAY.get(tu, tu):<20} {count}x  {bar}\n"
                )
            self.txtbox_log.insert("end", "\n─── Câu đã ghép ───\n")
            self.txtbox_log.insert(
                "end",
                " ".join([LABEL_DISPLAY.get(t, t) for t in self.cau_hien_tai]) or "..."
            )
            self.txtbox_log.configure(state="disabled")

    def _log(self, text: str):
        # Chỉ log thông báo hệ thống ra textbox khi không có "• " (từ nhận diện)
        if not text.startswith("•"):
            self.txtbox_log.configure(state="normal")
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            self.txtbox_log.insert("end", f"[{ts}] {text}\n")
            self.txtbox_log.see("end")
            self.txtbox_log.configure(state="disabled")


# ============================================================
if __name__ == "__main__":
    app = SignLanguageApp()
    app.mainloop()
