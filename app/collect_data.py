"""
手部数据采集工具（与识别一致的 MediaPipe Hands）
输入词汇 → 确认 → 录制手部关键点序列
"""
import cv2
import mediapipe as mp
import numpy as np
import os
import json

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "sequences")
os.makedirs(DATA_DIR, exist_ok=True)

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "collect_config.json")

DEFAULT_CONFIG = {
    "vocab": ["A", "B", "C", "D"],
    "seq_len": 30,
    "target_per_word": 20,
}


def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_CONFIG.copy()


def save_config(config):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def get_existing_counts():
    counts = {}
    if os.path.exists(DATA_DIR):
        for f in os.listdir(DATA_DIR):
            if f.endswith('.npy'):
                word = f.rsplit('_', 1)[0]
                counts[word] = counts.get(word, 0) + 1
    return counts


def normalize_landmarks(lm):
    """归一化：以手腕为原点，按手掌大小缩放"""
    pts = np.array([[p.x, p.y, p.z] for p in lm])
    pts = pts - pts[0]
    h = np.linalg.norm(pts[9])
    if h > 0:
        pts = pts / h
    return pts.flatten()


def record_word(word, seq_len, target, hands, cap):
    """录制一个词的多个样本"""
    mp_draw = mp.solutions.drawing_utils

    existing = [f for f in os.listdir(DATA_DIR)
                if f.startswith(word + '_') and f.endswith('.npy')]
    start_count = len(existing)
    count = start_count
    recording = False
    current_seq = []

    print(f"\n{'='*50}")
    print(f"录制: {word}")
    print(f"已有: {start_count}, 目标: {target}")
    print(f"空格=开始录制, q=跳过该词")
    print(f"{'='*50}")

    while count < target:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # 画手部关键点
        if results.multi_hand_landmarks:
            for hl in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hl, mp.solutions.hands.HAND_CONNECTIONS)

        # 录制中
        if recording and results.multi_hand_landmarks:
            hand_lm = results.multi_hand_landmarks[0].landmark
            features = normalize_landmarks(hand_lm)
            current_seq.append(features)
            frame_num = len(current_seq)

            if frame_num % 5 == 0:
                print(f"  帧 {frame_num}/{seq_len}")

            if frame_num >= seq_len:
                seq_file = os.path.join(DATA_DIR, f"{word}_{count:03d}.npy")
                np.save(seq_file, np.array(current_seq))
                current_seq = []
                recording = False
                count += 1
                print(f"  [保存] {word}_{count-1:03d}.npy ({count}/{target})")

        # UI
        status = f"REC {len(current_seq)}/{seq_len}" if recording else f"READY ({count}/{target})"
        color = (0, 0, 255) if recording else (0, 255, 0)
        cv2.putText(frame, f"Word: {word}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, status, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, "SPACE=record  Q=skip", (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        if recording:
            progress = len(current_seq) / seq_len
            bar_w = int(frame.shape[1] * 0.8)
            bar_x = int(frame.shape[1] * 0.1)
            bar_y = frame.shape[0] - 40
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 12), (50, 50, 50), -1)
            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x + int(bar_w * progress), bar_y + 12), (0, 0, 255), -1)

        cv2.imshow("Hand Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 32 and not recording:
            recording = True
            current_seq = []
            print(f"\n  [录制中] {word} ...")

    return count - start_count


def main():
    config = load_config()
    vocab = config["vocab"]
    seq_len = config["seq_len"]
    target = config["target_per_word"]

    print("=" * 60)
    print("手部数据采集工具（MediaPipe Hands）")
    print("=" * 60)
    print(f"词汇表: {', '.join(vocab)}")
    print(f"每词目标: {target} 个样本, 每样本: {seq_len} 帧")
    print()

    counts = get_existing_counts()
    print("已有样本:")
    for w in vocab:
        c = counts.get(w, 0)
        status = "✓" if c >= target else f"({c}/{target})"
        print(f"  {w}: {status}")
    print()

    while True:
        print("操作:")
        print("  1. 输入新词汇")
        print("  2. 开始采集（按词汇表顺序）")
        print("  3. 采集指定词汇")
        print("  4. 查看统计")
        print("  5. 退出")
        print()

        choice = input("请选择 (1-5): ").strip()

        if choice == "1":
            new_word = input("输入新词汇: ").strip()
            if new_word and new_word not in vocab:
                vocab.append(new_word)
                config["vocab"] = vocab
                save_config(config)
                print(f"已添加: {new_word}")
            elif new_word in vocab:
                print(f"'{new_word}' 已在词汇表中")
            print()

        elif choice == "2":
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("无法打开摄像头")
                continue

            with mp.solutions.hands.Hands(
                static_image_mode=False, max_num_hands=1,
                min_detection_confidence=0.7, min_tracking_confidence=0.5
            ) as hands:
                for word in vocab:
                    current_count = counts.get(word, 0)
                    if current_count >= target:
                        print(f"[跳过] {word} 已有 {current_count} 个样本")
                        continue
                    remaining = target - current_count
                    recorded = record_word(word, seq_len, remaining, hands, cap)
                    print(f"[完成] {word}: 新增 {recorded} 个样本")

            cap.release()
            cv2.destroyAllWindows()
            counts = get_existing_counts()
            print()

        elif choice == "3":
            word = input("输入词汇: ").strip()
            if word not in vocab:
                vocab.append(word)
                config["vocab"] = vocab
                save_config(config)

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("无法打开摄像头")
                continue

            current_count = counts.get(word, 0)
            remaining = max(target - current_count, 1)

            with mp.solutions.hands.Hands(
                static_image_mode=False, max_num_hands=1,
                min_detection_confidence=0.7, min_tracking_confidence=0.5
            ) as hands:
                recorded = record_word(word, seq_len, remaining, hands, cap)
                print(f"[完成] {word}: 新增 {recorded} 个样本")

            cap.release()
            cv2.destroyAllWindows()
            counts = get_existing_counts()
            print()

        elif choice == "4":
            counts = get_existing_counts()
            print(f"\n词汇表 ({len(vocab)} 个词):")
            total = 0
            for w in vocab:
                c = counts.get(w, 0)
                status = "✓" if c >= target else f"({c}/{target})"
                print(f"  {w}: {c} 个样本 {status}")
                total += c
            print(f"\n总计: {total} 个样本")
            print()

        elif choice == "5":
            print("退出")
            break
        else:
            print("无效选择\n")


if __name__ == "__main__":
    main()
