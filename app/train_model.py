"""
训练手部识别模型
从 MediaPipe Hands 采集的数据训练（63维/帧）
"""
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

SEQ_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "sequences")
MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "gestures", "gesture_model.pkl")


def augment(flat_data, n=4):
    augmented = [flat_data]
    for _ in range(n):
        noise = np.random.normal(0, 0.015, flat_data.shape)
        augmented.append(flat_data + noise)
    return augmented


def main():
    print("加载手部序列数据...")
    sequences = []
    labels = []

    for fname in sorted(os.listdir(SEQ_DIR)):
        if not fname.endswith('.npy'):
            continue
        label = fname.rsplit('_', 1)[0]
        seq = np.load(os.path.join(SEQ_DIR, fname))

        # 检查数据格式
        if len(seq.shape) == 2 and seq.shape[1] >= 63:
            # 每帧至少63维（手部），取前63维
            hand_seq = seq[:, :63].flatten()
        else:
            print(f"跳过 {fname}，格式不对: {seq.shape}")
            continue

        for aug in augment(hand_seq):
            sequences.append(aug)
            labels.append(label)

    X = np.array(sequences)
    y = np.array(labels)

    print(f"总样本数: {len(X)}")
    print(f"特征维度: {X.shape[1]} (63/帧 × {X.shape[1]//63}帧)")
    print(f"类别: {sorted(set(y))}")
    print("每类样本数:")
    for cls, cnt in sorted(zip(*np.unique(y, return_counts=True))):
        print(f"  {cls}: {cnt}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n训练 Random Forest...")
    model = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n准确率: {acc*100:.1f}%")

    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"5折交叉验证: {scores.mean()*100:.1f}% (+/- {scores.std()*100:.1f}%)")

    print("\n分类报告:")
    print(classification_report(y_test, y_pred))

    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    print(f"模型已保存: {MODEL_FILE}")
    print(f"文件大小: {os.path.getsize(MODEL_FILE)/1024:.0f} KB")


if __name__ == "__main__":
    main()
