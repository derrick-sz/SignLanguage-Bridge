"""
从全身序列数据中提取手部关键点，训练手部识别模型
输入：sequences/ 中的全身数据（225维/帧）
输出：只用手部数据（126维/帧 = 左手63 + 右手63）训练的模型
"""
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

SEQ_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "sequences")
MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "gestures", "gesture_model.pkl")

# 身体33点×3 = 99, 左手21点×3 = 63, 右手21点×3 = 63
HAND_START = 99  # 从第99个特征开始是手部数据


def extract_hand_features(seq):
    """从全身序列中提取手部关键点（每帧126维 = 左手63 + 右手63）"""
    hand_frames = []
    for frame in seq:
        # 左手 + 右手
        left_hand = frame[99:162]   # 63维
        right_hand = frame[162:225]  # 63维
        hand_frames.append(np.concatenate([left_hand, right_hand]))
    return np.array(hand_frames).flatten()


def augment(flat_data, n=4):
    """数据增强：加噪声"""
    augmented = [flat_data]
    for _ in range(n):
        noise = np.random.normal(0, 0.015, flat_data.shape)
        augmented.append(flat_data + noise)
    return augmented


def main():
    print("从全身数据中提取手部特征...")
    sequences = []
    labels = []

    for fname in sorted(os.listdir(SEQ_DIR)):
        if not fname.endswith('.npy'):
            continue
        label = fname.rsplit('_', 1)[0]
        seq = np.load(os.path.join(SEQ_DIR, fname))

        # 提取手部特征
        hand_flat = extract_hand_features(seq)

        # 原始 + 增强
        for aug in augment(hand_flat):
            sequences.append(aug)
            labels.append(label)

    X = np.array(sequences)
    y = np.array(labels)

    print(f"总样本数: {len(X)}")
    print(f"特征维度: {X.shape[1]} (手部: 左手63 + 右手63 = 126/帧 × {X.shape[1]//126}帧)")
    print(f"类别: {sorted(set(y))}")
    print("每类样本数:")
    for cls, cnt in sorted(zip(*np.unique(y, return_counts=True))):
        print(f"  {cls}: {cnt}")

    # 划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 训练
    print("\n训练 Random Forest...")
    model = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # 评估
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n准确率: {acc*100:.1f}%")

    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"5折交叉验证: {scores.mean()*100:.1f}% (+/- {scores.std()*100:.1f}%)")

    print("\n分类报告:")
    print(classification_report(y_test, y_pred))

    # 保存
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    print(f"模型已保存: {MODEL_FILE}")
    print(f"文件大小: {os.path.getsize(MODEL_FILE)/1024:.0f} KB")


if __name__ == "__main__":
    main()
