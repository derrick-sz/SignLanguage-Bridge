"""
中巴手语桥 - 桌面版
方案A: 摄像头实时识别手势 → 中文 + 葡萄牙语
方案B: 输入文字 → 播放骨架动画
"""
import cv2
import mediapipe as mp
import numpy as np
import os
import joblib
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from collections import deque, Counter
import threading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "..", "data", "gestures", "gesture_model.pkl")
SEQ_DIR = os.path.join(BASE_DIR, "..", "data", "sequences_full")

TRANSLATIONS = {
    "A": "A", "B": "B", "C": "C", "D": "D",
}

# 骨架渲染
POSE_CONNS = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24)]
HAND_CONNS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),
]


def render_skeleton(features):
    W, H = 400, 500
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    pose = features[:99].reshape(33, 3)
    lh = features[99:162].reshape(21, 3)
    rh = features[162:225].reshape(21, 3)
    all_pts = [[p[0], p[1]] for p in np.vstack([pose, lh, rh]) if p[0] > -900]
    if not all_pts:
        return canvas
    all_pts = np.array(all_pts)
    xmin, ymin = all_pts.min(axis=0)
    xmax, ymax = all_pts.max(axis=0)
    xr = max(xmax - xmin, 1)
    yr = max(ymax - ymin, 1)

    def px(x, y, m=0.12):
        return (int((m + (x - xmin) / xr * (1 - 2*m)) * W),
                int((m + (y - ymin) / yr * (1 - 2*m)) * H))

    def ok(p):
        return p[0] > -900

    RED = (0, 0, 255)

    for s, e in POSE_CONNS:
        if ok(pose[s]) and ok(pose[e]):
            cv2.line(canvas, px(*pose[s][:2]), px(*pose[e][:2]), RED, 3)
    for p in pose:
        if ok(p):
            cv2.circle(canvas, px(p[0], p[1]), 5, RED, -1)

    if ok(pose[0]):
        n = px(*pose[0][:2])
        if ok(pose[11]) and ok(pose[12]):
            ls = px(*pose[11][:2])
            rs = px(*pose[12][:2])
            mid = ((ls[0]+rs[0])//2, (ls[1]+rs[1])//2)
            r = int(np.sqrt((n[0]-mid[0])**2 + (n[1]-mid[1])**2) * 0.7)
            cv2.circle(canvas, n, r, RED, 2)
            cv2.line(canvas, (n[0], n[1]+r), mid, RED, 2)
        else:
            cv2.circle(canvas, n, 35, RED, 2)

    for hand in [lh, rh]:
        for s, e in HAND_CONNS:
            if s < len(hand) and e < len(hand) and ok(hand[s]) and ok(hand[e]):
                cv2.line(canvas, px(*hand[s][:2]), px(*hand[e][:2]), RED, 2)
        for p in hand:
            if ok(p):
                cv2.circle(canvas, px(p[0], p[1]), 3, RED, -1)

    return canvas


def normalize_landmarks(lm):
    pts = np.array([[p.x, p.y, p.z] for p in lm])
    pts = pts - pts[0]
    h = np.linalg.norm(pts[9])
    if h > 0:
        pts = pts / h
    return pts.flatten().reshape(1, -1)


class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("中巴手语桥 | Sign Language Bridge")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f0f0f0")

        # 加载模型
        self.ml_model = None
        if os.path.exists(MODEL_FILE):
            raw = joblib.load(MODEL_FILE)
            self.ml_model = raw['model'] if isinstance(raw, dict) else raw

        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, max_num_hands=1,
            min_detection_confidence=0.6, min_tracking_confidence=0.5
        )

        # 状态
        self.cam_on = False
        self.cap = None
        self.history = deque(maxlen=10)
        self.current_gesture = None
        self.hand_buffer = deque(maxlen=30)  # 30帧手部特征缓冲
        self.frame_count = 0

        self.build_ui()

    def build_ui(self):
        # 标题
        title = tk.Label(self.root, text="🤝 中巴手语桥 | Ponte de Sinais China-Brasil",
                         font=("Arial", 18, "bold"), bg="#DE2910", fg="white", pady=10)
        title.pack(fill=tk.X)

        # 主框架
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左侧：摄像头
        left_frame = tk.LabelFrame(main_frame, text="📡 方案A：摄像头识别", font=("Arial", 12, "bold"),
                                   bg="white", padx=5, pady=5)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.cam_label = tk.Label(left_frame, bg="black", width=50, height=25)
        self.cam_label.pack(fill=tk.BOTH, expand=True)

        cam_ctrl = tk.Frame(left_frame, bg="white")
        cam_ctrl.pack(fill=tk.X, pady=5)

        self.cam_btn = tk.Button(cam_ctrl, text="▶ 启动识别", font=("Arial", 11, "bold"),
                                 bg="#009739", fg="white", command=self.toggle_camera)
        self.cam_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        # 参数调节
        params_frame = tk.LabelFrame(left_frame, text="⚙️ 参数调节", font=("Arial", 10),
                                     bg="white", padx=5, pady=5)
        params_frame.pack(fill=tk.X, pady=5)

        # 置信度阈值
        tk.Label(params_frame, text="置信度阈值:", font=("Arial", 9), bg="white").grid(row=0, column=0, sticky=tk.W)
        self.conf_var = tk.DoubleVar(value=0.25)
        self.conf_slider = tk.Scale(params_frame, from_=0.1, to=1.0, resolution=0.05,
                                     orient=tk.HORIZONTAL, variable=self.conf_var, bg="white")
        self.conf_slider.grid(row=0, column=1, sticky=tk.EW, padx=5)

        # 分类频率
        tk.Label(params_frame, text="分类频率:", font=("Arial", 9), bg="white").grid(row=1, column=0, sticky=tk.W)
        self.classify_var = tk.IntVar(value=1)
        self.classify_slider = tk.Scale(params_frame, from_=1, to=10, orient=tk.HORIZONTAL,
                                         variable=self.classify_var, bg="white")
        self.classify_slider.grid(row=1, column=1, sticky=tk.EW, padx=5)

        # 稳定阈值
        tk.Label(params_frame, text="稳定阈值:", font=("Arial", 9), bg="white").grid(row=2, column=0, sticky=tk.W)
        self.stable_var = tk.IntVar(value=2)
        self.stable_slider = tk.Scale(params_frame, from_=1, to=10, orient=tk.HORIZONTAL,
                                       variable=self.stable_var, bg="white")
        self.stable_slider.grid(row=2, column=1, sticky=tk.EW, padx=5)

        params_frame.columnconfigure(1, weight=1)

        # 右侧：结果 + 骨架动画
        right_frame = tk.Frame(main_frame, bg="#f0f0f0")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # 识别结果
        result_frame = tk.LabelFrame(right_frame, text="识别结果", font=("Arial", 12, "bold"),
                                     bg="white", padx=10, pady=10)
        result_frame.pack(fill=tk.X, pady=(0, 10))

        self.result_zh = tk.Label(result_frame, text="等待识别...", font=("Arial", 24, "bold"),
                                  bg="white", fg="#333")
        self.result_zh.pack()

        self.result_pt = tk.Label(result_frame, text="", font=("Arial", 16),
                                  bg="white", fg="#009739")
        self.result_pt.pack()

        # 骨架动画
        anim_frame = tk.LabelFrame(right_frame, text="🤟 方案B：骨架动画", font=("Arial", 12, "bold"),
                                   bg="white", padx=5, pady=5)
        anim_frame.pack(fill=tk.BOTH, expand=True)

        self.anim_label = tk.Label(anim_frame, bg="black", width=40, height=18)
        self.anim_label.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        anim_ctrl = tk.Frame(anim_frame, bg="white")
        anim_ctrl.pack(fill=tk.X)

        tk.Label(anim_ctrl, text="词汇:", font=("Arial", 10), bg="white").pack(side=tk.LEFT)
        self.word_entry = tk.Entry(anim_ctrl, font=("Arial", 12), width=10)
        self.word_entry.pack(side=tk.LEFT, padx=5)
        self.word_entry.insert(0, "A")

        self.play_btn = tk.Button(anim_ctrl, text="▶ 播放", font=("Arial", 10, "bold"),
                                  bg="#0066cc", fg="white", command=self.play_animation)
        self.play_btn.pack(side=tk.LEFT, padx=5)

        # 底部状态栏
        self.status_bar = tk.Label(self.root, text="就绪 | 词汇表: A B C D",
                                   font=("Arial", 9), bg="#ddd", anchor=tk.W, padx=10)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def toggle_camera(self):
        if self.cam_on:
            self.cam_on = False
            self.cam_btn.config(text="▶ 启动识别", bg="#009739")
            if self.cap:
                self.cap.release()
                self.cap = None
            self.cam_label.config(image="", bg="black")
            self.status_bar.config(text="摄像头已关闭")
        else:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("错误", "无法打开摄像头")
                return
            self.cam_on = True
            self.cam_btn.config(text="⏹ 停止识别", bg="#cc0000")
            self.status_bar.config(text="摄像头运行中...")
            self.update_camera()

    def update_camera(self):
        if not self.cam_on:
            return

        self.frame_count += 1
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        # 画手部关键点
        if results.multi_hand_landmarks:
            for hl in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hl, self.mp_hands.HAND_CONNECTIONS)

            # ML识别（使用滑块参数）
            classify_every = self.classify_var.get()
            if self.frame_count % classify_every == 0:
                gesture = self.ml_detect(results.multi_hand_landmarks[0].landmark)
                if gesture:
                    self.history.append(gesture)
                    stable_count = self.stable_var.get()
                    if len(self.history) >= stable_count:
                        mc = Counter(self.history).most_common(1)[0]
                        if mc[1] >= stable_count:
                            self.current_gesture = mc[0]

        # 显示识别结果
        if self.current_gesture:
            zh = self.current_gesture
            pt = TRANSLATIONS.get(zh, zh)
            self.result_zh.config(text=zh)
            self.result_pt.config(text=pt)
            self.status_bar.config(text=f"识别到: {zh} | {pt}")

        # 显示摄像头画面
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img = img.resize((480, 360), Image.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        self.cam_label.imgtk = imgtk
        self.cam_label.config(image=imgtk)

        self.root.after(30, self.update_camera)

    def ml_detect(self, lm):
        if self.ml_model is None or lm is None or len(lm) < 21:
            return None

        # 提取当前帧手部特征（63维）
        hand_feat = normalize_landmarks(lm).flatten()
        self.hand_buffer.append(hand_feat)

        # 需要30帧才能预测
        if len(self.hand_buffer) < 30:
            return None

        # 组合30帧（1890维）
        seq = np.array(list(self.hand_buffer)).flatten().reshape(1, -1)
        pred = self.ml_model.predict(seq)[0]
        conf = float(np.max(self.ml_model.predict_proba(seq)[0]))

        threshold = self.conf_var.get()
        if conf >= threshold and pred in TRANSLATIONS:
            return pred
        return None

    def play_animation(self):
        word = self.word_entry.get().strip()
        if not word:
            messagebox.showwarning("提示", "请输入词汇")
            return

        seq_file = os.path.join(SEQ_DIR, f"{word}_000.npy")
        if not os.path.exists(seq_file):
            messagebox.showwarning("提示", f"未找到 '{word}' 的数据")
            return

        data = np.load(seq_file)
        self.status_bar.config(text=f"播放: {word} ({len(data)} 帧)")
        self.play_btn.config(state=tk.DISABLED)

        def animate(i):
            if i >= len(data):
                self.play_btn.config(state=tk.NORMAL)
                self.status_bar.config(text="播放完成")
                return

            sk = render_skeleton(data[i])
            sk = cv2.cvtColor(sk, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(sk)
            img = img.resize((320, 400), Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            self.anim_label.imgtk = imgtk
            self.anim_label.config(image=imgtk)
            self.root.after(80, animate, i + 1)

        animate(0)

    def on_close(self):
        self.cam_on = False
        if self.cap:
            self.cap.release()
        self.hands.close()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
