import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from PIL import Image

# Cố định seed và tạo thư mục lưu biểu đồ
tf.random.set_seed(42)
os.makedirs("results", exist_ok=True)

CLASS_NAMES = ["Airplane", "Automobile", "Bird", "Cat", "Deer",
               "Dog", "Frog", "Horse", "Ship", "Truck"]


# 1. TẢI VÀ CHUẨN HÓA DỮ LIỆU CIFAR-10 (Ảnh màu 32x32x3)
(x_train_full, y_train_full), (x_test, y_test) = cifar10.load_data()
y_train_full = y_train_full.flatten()
y_test = y_test.flatten()

# Xuất Hình: 10 ảnh mẫu đại diện cho 10 lớp
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    idx = (y_train_full == i).nonzero()[0][0]
    ax.imshow(x_train_full[idx])
    ax.set_title(f"{i}: {CLASS_NAMES[i]}", fontsize=9)
    ax.axis("off")
fig.suptitle("Sample Input Images from CIFAR-10 Dataset (32x32x3)", fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig("results/fig1_input_samples.png", dpi=300)
plt.close(fig)

# Chuẩn hóa pixel về [0, 1] và chia Train (40,000) / Validation (10,000)
x_train_full = x_train_full.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_val, y_val = x_train_full[:10000], y_train_full[:10000]
x_train, y_train = x_train_full[10000:], y_train_full[10000:]


# 2. XÂY DỰNG 3 MÔ HÌNH
# Mô hình 1: Baseline CNN chưa có kỹ thuật giảm Overfitting
def build_baseline():
    model = models.Sequential([
        layers.Input(shape=(32, 32, 3)),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# Mô hình 2: Regularized CNN (Data Augmentation + Dropout)
def build_regularized():
    model = models.Sequential([
        layers.Input(shape=(32, 32, 3)),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# Mô hình 3 (Mở rộng): Improved CNN (3 Conv Blocks + BatchNormalization)
def build_improved():
    model = models.Sequential([
        layers.Input(shape=(32, 32, 3)),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# 3. HUẤN LUYỆN CẢ 3 MÔ HÌNH
EPOCHS = 15
BATCH_SIZE = 128

models_dict = {
    "1. Baseline CNN": build_baseline(),
    "2. Regularized CNN": build_regularized(),
    "3. Improved CNN (BatchNorm)": build_improved()
}

histories = {}
for name, model in models_dict.items():
    print(f"\n--- Đang huấn luyện: {name} ---")
    hist = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                     validation_data=(x_val, y_val), verbose=1)
    histories[name] = hist.history

# 4. IN BẢNG SỐ LIỆU HOÀN CHỈNH RA TERMINAL (CHO BẢNG 2.3 VÀ BẢNG 2.4)
print("\n" + "="*98)
print(f"{'Model':<28} | {'Train Acc':<9} | {'Val Acc':<8} | {'Test Acc':<8} | {'Test Loss':<9} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8}")
print("-" * 98)

for name, model in models_dict.items():
    h = histories[name]
    train_acc = h["accuracy"][-1] * 100
    val_acc = h["val_accuracy"][-1] * 100
    
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")
    
    print(f"{name:<28} | {train_acc:>8.2f}% | {val_acc:>7.2f}% | {test_acc*100:>7.2f}% | {test_loss:>9.4f} | {prec:>9.4f} | {rec:>8.4f} | {f1:>8.4f}")

print("="*98)

# 5. XUẤT BIỂU ĐỒ
# --- Hình 2: Learning Curves Model 1 vs Model 2 ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
for name in ["1. Baseline CNN", "2. Regularized CNN"]:
    h = histories[name]
    ax1.plot(h["val_loss"], label=name, linewidth=2)
    ax2.plot(h["val_accuracy"], label=name, linewidth=2)
ax1.set_title("Validation Loss: Baseline vs. Regularized")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend(); ax1.grid(True, alpha=0.3)
ax2.set_title("Validation Accuracy: Baseline vs. Regularized")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.legend(); ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/fig2_part_b_curves.png", dpi=300)
plt.close()

# --- Hình 3: Confusion Matrix Model 2 ---
y_pred_m2 = np.argmax(models_dict["2. Regularized CNN"].predict(x_test, verbose=0), axis=1)
plt.figure(figsize=(8.5, 6.5))
sns.heatmap(confusion_matrix(y_test, y_pred_m2), annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("Confusion Matrix - Regularized CNN (Test Set)")
plt.xlabel("Predicted Label"); plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("results/fig3_part_b_cm.png", dpi=300)
plt.close()

# --- Hình 4: Learning Curves Model 2 vs Model 3 ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
for name in ["2. Regularized CNN", "3. Improved CNN (BatchNorm)"]:
    h = histories[name]
    ax1.plot(h["val_loss"], label=name, linewidth=2)
    ax2.plot(h["val_accuracy"], label=name, linewidth=2)
ax1.set_title("Validation Loss: Regularized vs. Improved CNN")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend(); ax1.grid(True, alpha=0.3)
ax2.set_title("Validation Accuracy: Regularized vs. Improved CNN")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.legend(); ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/fig4_part_c_curves.png", dpi=300)
plt.close()

# --- Hình 5: Confusion Matrix Model 3 ---
best_model = models_dict["3. Improved CNN (BatchNorm)"]
y_pred_m3 = np.argmax(best_model.predict(x_test, verbose=0), axis=1)
print("\nBáo cáo phân loại chi tiết trên tập Test (Model 3):")
print(classification_report(y_test, y_pred_m3, target_names=CLASS_NAMES))

plt.figure(figsize=(8.5, 6.5))
sns.heatmap(confusion_matrix(y_test, y_pred_m3), annot=True, fmt="d", cmap="Greens",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("Confusion Matrix - Improved CNN with BatchNorm (Test Set)")
plt.xlabel("Predicted Label"); plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("results/fig5_part_c_cm.png", dpi=300)
plt.close()

print("\nHoàn tất! Đã lưu đủ 5 biểu đồ (fig1 -> fig5) trong thư mục 'results/'.")