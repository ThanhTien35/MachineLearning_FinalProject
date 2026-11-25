# Introduction to Machine Learning - Final Project

## 1. Cấu trúc project

```text
MachineLearning_FinalProject/
├── Question1_FNN/
│   ├── data.py
│   ├── fnn.py
│   ├── techniques.py
│   ├── main.py
│   ├── requirements.txt
│   └── results/
├── Question2_CNN/
│   ├── cnn_cifar10.py
│   ├── requirements.txt
│   └── results/
├── .gitignore
└── README.md
```

## 2. Question 1 - Feedforward Neural Network (FNN)

Xây dựng mạng FNN từ đầu bằng NumPy để giải quyết bài toán hồi quy trên bộ dữ liệu Diabetes, kết hợp Early Stopping và L2 Regularization nhằm hạn chế Overfitting.

### 2.1. Câu 1a - Cơ sở lý thuyết

- Kiến trúc FNN với nhiều tầng ẩn.
- Forward Propagation.
- Mean Squared Error (MSE).
- Backpropagation và Chain Rule.
- Chứng minh công thức gradient của Weight và Bias.
- Thuật toán Gradient Descent và mã giả.

**File tham khảo:** `fnn.py`.

Phần lý thuyết, chứng minh công thức và mã giả được trình bày trong báo cáo.

### 2.2. Câu 1b - Cài đặt ứng dụng

Tự cài đặt mạng FNN bằng Python và NumPy, không sử dụng framework xây dựng mạng neural hoặc tự động tính gradient.

**Dataset:**

- Diabetes Dataset.
- Số lượng mẫu: 442.
- Số lượng đặc trưng: 10.
- Training Set: 70%.
- Validation Set: 15%.
- Testing Set: 15%.

**Cấu hình mô hình:**

| Parameter | Value |
|---|---|
| Architecture | 10-64-32-1 |
| Hidden Layers | 2 |
| Hidden Activation | ReLU |
| Output Activation | Linear |
| Loss Function | MSE |
| Optimizer | Gradient Descent |
| Learning Rate | 0.01 |
| Maximum Epochs | 2000 |
| Random Seed | 42 |

**Source code:**

| File | Nội dung |
|---|---|
| `data.py` | Tải dữ liệu, chia tập dữ liệu và chuẩn hóa |
| `fnn.py` | Khởi tạo tham số, Forward Propagation, Backpropagation và cập nhật tham số |
| `techniques.py` | Quản lý quá trình huấn luyện |
| `main.py` | Chạy thực nghiệm và đánh giá mô hình |

### 2.3. Câu 1c - Early Stopping

Cài đặt kỹ thuật Early Stopping nhằm hạn chế Overfitting dựa trên Validation Loss.

**Nội dung thực hiện:**

- Theo dõi Validation Loss sau mỗi epoch.
- Sử dụng patience và min_delta.
- Dừng huấn luyện khi mô hình không còn cải thiện.
- Lưu và khôi phục bộ trọng số và bias tốt nhất.

**Cấu hình:**

| Parameter | Value |
|---|---|
| Patience | 50 |
| Min Delta | 0.0001 |
| Monitor | Validation Loss |
| Restore Best Parameters | True |

**Source code:** `techniques.py`, `main.py`.

### 2.4. Câu 1d - L2 Regularization

Tích hợp L2 Regularization vào hàm mục tiêu và quá trình Backpropagation của FNN.

**Nội dung thực hiện:**

- Kết hợp MSE và L2 Regularization.
- Bổ sung thành phần L2 vào gradient của Weight.
- Không áp dụng L2 lên Bias.
- Thử nghiệm và lựa chọn hệ số Lambda dựa trên tập Validation.

**Cấu hình:**

| Parameter | Value |
|---|---|
| Lambda Candidates | 0.001, 0.01, 0.03, 0.1, 0.3 |
| Selected Lambda | 0.1 |

**Source code:** `fnn.py`, `techniques.py`, `main.py`.

### 2.5. Câu 1e - Thực nghiệm và đánh giá

Huấn luyện và so sánh bốn cấu hình FNN:

| Model | Early Stopping | L2 Regularization |
|---|---|---|
| Baseline FNN | No | No |
| FNN + Early Stopping | Yes | No |
| FNN + L2 | No | Yes |
| FNN + Early Stopping + L2 | Yes | Yes |

**Chỉ số đánh giá:**

- Mean Squared Error (MSE).
- Root Mean Squared Error (RMSE).
- Mean Absolute Error (MAE).

**Nội dung phân tích:**

- So sánh Training Loss và Validation Loss.
- So sánh kết quả trên tập Testing.
- Phân tích hiện tượng Overfitting.
- Đánh giá khả năng tổng quát hóa của từng mô hình.

**Source code:** `main.py`.

**Kết quả:** `Question1_FNN/results/`.

### 2.6. Hướng dẫn chạy Câu 1

Mở Terminal tại thư mục gốc của project:

```bash
cd Question1_FNN
python -m pip install -r requirements.txt
python main.py
```

Kết quả thực nghiệm được lưu trong thư mục `results/`.

---

## 3. Question 2 - Convolutional Neural Network (CNN)

Xây dựng mạng CNN bằng TensorFlow/Keras để giải quyết bài toán phân loại hình ảnh trên bộ dữ liệu CIFAR-10.

### 3.1. Câu 2a - Cơ sở lý thuyết

- Kiến trúc Convolutional Neural Network.
- Convolutional Layer.
- Pooling Layer.
- Activation Function.
- Fully Connected Layer.
- Output Layer và Softmax.
- Quy trình Training và Inference.
- So sánh CNN với FNN truyền thống.

**Source code tham khảo:**

- `cnn_cifar10.py`: Phần 1 - Tải và chuẩn hóa dữ liệu.
- `cnn_cifar10.py`: Hàm `build_baseline()`.

**Hình ảnh minh họa:**

`results/fig1_input_samples.png`

Phần giải thích lý thuyết, công thức toán học và so sánh được trình bày trong báo cáo.

### 3.2. Câu 2b - Thực nghiệm và phân tích

Xây dựng mô hình Baseline CNN và áp dụng hai kỹ thuật giảm Overfitting:

- Data Augmentation.
- Dropout.

**Dataset:**

| Parameter | Value |
|---|---|
| Dataset | CIFAR-10 |
| Input Shape | 32 x 32 x 3 |
| Number of Classes | 10 |
| Training Set | 40,000 images |
| Validation Set | 10,000 images |
| Testing Set | 10,000 images |

**Cấu hình huấn luyện:**

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |
| Epochs | 15 |
| Batch Size | 128 |

**Các mô hình thực nghiệm:**

| Model | Data Augmentation | Dropout |
|---|---|---|
| Baseline CNN | No | No |
| Regularized CNN | Yes | Yes |

**Source code trong `cnn_cifar10.py`:**

| Hàm / Phần code | Nội dung |
|---|---|
| `build_baseline()` | Xây dựng CNN cơ bản |
| `build_regularized()` | CNN kết hợp Data Augmentation và Dropout |
| Phần 3 | Huấn luyện các mô hình |
| Phần 4 | Tính toán và so sánh các chỉ số đánh giá |
| Phần 5 | Trực quan hóa kết quả |

**Hình ảnh kết quả:**

| File | Nội dung |
|---|---|
| `fig2_part_b_curves.png` | So sánh Validation Loss và Validation Accuracy của Baseline CNN và Regularized CNN |
| `fig3_part_b_cm.png` | Confusion Matrix của Regularized CNN |

### 3.3. Câu 2c - Nghiên cứu mở rộng

Xây dựng Improved CNN nhằm cải thiện hiệu quả phân loại của Regularized CNN.

**Các cải tiến:**

- Tăng số lượng Convolutional Blocks từ 2 lên 3.
- Bổ sung Batch Normalization.
- Tăng số lượng filters lên 32, 64 và 128.
- Tăng số lượng neurons của Dense Layer từ 128 lên 256.
- Tiếp tục sử dụng Data Augmentation và Dropout.

**Các mô hình so sánh:**

| Model | Conv Blocks | Batch Normalization | Dropout | Data Augmentation |
|---|---|---|---|---|
| Regularized CNN | 2 | No | Yes | Yes |
| Improved CNN | 3 | Yes | Yes | Yes |

**Source code trong `cnn_cifar10.py`:**

| Hàm / Phần code | Nội dung |
|---|---|
| `build_improved()` | Xây dựng mô hình CNN cải tiến |
| Phần 3 | Huấn luyện mô hình |
| Phần 4 | So sánh các chỉ số đánh giá |
| Phần 5 | Xuất biểu đồ và Confusion Matrix |
| `classification_report()` | Báo cáo kết quả phân loại chi tiết |

**Hình ảnh kết quả:**

| File | Nội dung |
|---|---|
| `fig4_part_c_curves.png` | So sánh Validation Loss và Validation Accuracy của Regularized CNN và Improved CNN |
| `fig5_part_c_cm.png` | Confusion Matrix của Improved CNN |

### 3.4. Các chỉ số đánh giá

- Training Accuracy.
- Validation Accuracy.
- Testing Accuracy.
- Testing Loss.
- Precision (Macro Average).
- Recall (Macro Average).
- F1-Score (Macro Average).
- Confusion Matrix.

Các chỉ số được hiển thị trên Terminal sau khi huấn luyện.

### 3.5. Hướng dẫn chạy Câu 2

Mở Terminal tại thư mục gốc của project:

```bash
cd Question2_CNN
python -m pip install -r requirements.txt
python cnn_cifar10.py
```

Bộ dữ liệu CIFAR-10 được tải tự động trong lần chạy đầu tiên.

Các biểu đồ kết quả được lưu trong thư mục `results/`.

---

## 4. Tổng hợp source code theo yêu cầu đề bài

| Câu | Nội dung | Source code tham khảo |
|---|---|---|
| 1a | FNN Theory | `fnn.py` |
| 1b | FNN Implementation | `data.py`, `fnn.py`, `techniques.py`, `main.py` |
| 1c | Early Stopping | `techniques.py`, `main.py` |
| 1d | L2 Regularization | `fnn.py`, `techniques.py`, `main.py` |
| 1e | Experiments and Evaluation | `main.py`, `results/` |
| 2a | CNN Theory | `cnn_cifar10.py` - `build_baseline()` |
| 2b | CNN and Overfitting Reduction | `cnn_cifar10.py` - `build_baseline()`, `build_regularized()` |
| 2c | Improved CNN | `cnn_cifar10.py` - `build_improved()` |

## 5. Lưu ý

- Khuyến nghị sử dụng Python 3.11.
- Cần kết nối Internet để cài đặt thư viện và tải dữ liệu.
- Mỗi câu có file `requirements.txt` riêng.
- Thời gian huấn luyện phụ thuộc vào cấu hình phần cứng.
- Không đưa môi trường ảo `.venv` và `__pycache__` lên GitHub.