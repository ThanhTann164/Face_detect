# Hệ Thống Nhận Diện Khuôn Mặt & Điều Khiển Cửa Tự Động

Hệ thống nhận diện khuôn mặt sử dụng MTCNN và FaceNet, tích hợp MQTT để điều khiển cửa tự động qua ESP32.

---

## 📋 Mục Lục

1. [Tổng Quan Hệ Thống](#tổng-quan-hệ-thống)
2. [Cài Đặt](#cài-đặt)
3. [Cách Sử Dụng](#cách-sử-dụng)
4. [Cấu Hình](#cấu-hình)
5. [API Endpoints](#api-endpoints)
6. [Thêm Người Mới](#thêm-người-mới)
7. [Cấu Trúc File](#cấu-trúc-file)
8. [Xử Lý Lỗi](#xử-lý-lỗi)

---

## 📋 Tổng Quan Hệ Thống

### Kiến Trúc Hệ Thống

Hệ thống gồm 4 thành phần chính:

1. **AI Server** (`face_rec_flask.py`) - Server Flask nhận diện khuôn mặt và gửi lệnh MQTT
2. **Camera Client** (`camera_client.py`) - Capture ảnh từ camera và gửi lên AI server
3. **Logic Service** (`logic_service.py`) - Xử lý điều kiện mở cửa (tùy chọn)
4. **ESP32** - Nhận lệnh MQTT và điều khiển servo mở cửa

### Quy Trình Hoạt Động

```
Camera → AI Server → Nhận Diện → MQTT → ESP32 → Mở Cửa
```

1. Camera capture ảnh và gửi lên AI server
2. AI server nhận diện khuôn mặt bằng FaceNet + SVM classifier
3. Nếu nhận diện thành công (probability > 0.8):
   - AI server gửi lệnh "OPEN" qua MQTT
   - ESP32 nhận lệnh và mở cửa (servo quay 90 độ)
   - Sau 7 giây, ESP32 tự đóng cửa

---

## 🔧 Cài Đặt

### Yêu Cầu Hệ Thống

- **Python**: 3.8 - 3.10 (không dùng Python 3.11+ vì TensorFlow chưa hỗ trợ)
- **TensorFlow**: 2.10.0
- **Thư viện**: Xem `requirements.txt`

### Bước 1: Cài Đặt Python

1. Tải Python 3.10 từ: https://www.python.org/downloads/
2. Cài đặt và tick "Add Python to PATH"

### Bước 2: Cài Đặt Thư Viện

```bash
cd D:\Face_Mi_AI\MiAI_FaceRecog_3
python -m pip install -r requirements.txt
```

**Lưu ý:** Nếu gặp lỗi cài TensorFlow:
```bash
pip install tensorflow==2.10.0
```

### Bước 3: Kiểm Tra Models

Đảm bảo có các file sau:
- `Models/20180402-114759/20180402-114759.pb` - FaceNet model
- `Models/facemodel_new.pkl` hoặc `Models/facemodel.pkl` - Classifier model

---

## 🚀 Cách Sử Dụng

### Cách 1: Chạy Đơn Giản (AI Server Tự Mở Cửa)

**Bước 1:** Khởi động AI Server
```bash
cd D:\Face_Mi_AI\MiAI_FaceRecog_3\src
python face_rec_flask.py
```

Bạn sẽ thấy:
```
Custom Classifier, Successfully loaded
Loading feature extraction model
✅ Đã kết nối tới MQTT broker
 * Running on http://0.0.0.0:8000
```

**Bước 2:** Chạy Camera Client (terminal mới)
```bash
cd D:\Face_Mi_AI\MiAI_FaceRecog_3\src
python camera_client.py
```

**Tùy chọn:**
```bash
# Dùng camera ID khác
python camera_client.py --camera 1

# Chỉ định server AI (nếu chạy trên máy khác)
python camera_client.py --server http://192.168.1.100:8000

# Thay đổi interval gửi ảnh (giây)
python camera_client.py --interval 3

# Gửi tất cả ảnh (không chỉ khi có mặt)
python camera_client.py --always-send
```

**Bước 3:** Kiểm tra hoạt động
- Camera sẽ hiển thị preview
- Khi nhận diện được khuôn mặt (probability > 0.8), AI server sẽ tự động gửi lệnh "OPEN" tới ESP32
- ESP32 sẽ mở cửa và tự đóng sau 7 giây
- Xem log trên Serial Monitor ESP32 để xác nhận

### Cách 2: Sử Dụng Web Interface

**Truy cập các trang web:**

- **Trang chủ**: `http://localhost:8000/`
- **Panel chụp ảnh**: `http://localhost:8000/capture`
- **Panel tự động mở cửa**: `http://localhost:8000/auto-door`
- **Panel điều khiển thủ công**: `http://localhost:8000/manual-control`
- **Control panel**: `http://localhost:8000/control`

---

## ⚙️ Cấu Hình

### Cấu Hình MQTT

Sửa trong `src/face_rec_flask.py`:

```python
MQTT_SERVER = "7a28606d7a234d84a5035fa5e28698a3.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_USER = "nguyenluc0112"
MQTT_PASSWORD = "buithanhTan@123"
MQTT_TOPIC_DOOR_CMD = "door/cmd"
MQTT_TLS_INSECURE = True  # False khi deploy thực tế
```

### Thay Đổi Ngưỡng Nhận Diện

Sửa trong `src/face_rec_flask.py`:

```python
MIN_CONFIDENCE_THRESHOLD = 0.30  # Ngưỡng probability (0.0 - 1.0)
```

Hoặc trong code xử lý:
```python
if best_class_probabilities > 0.8:  # Thay 0.8 bằng giá trị khác
```

### Thay Đổi Danh Sách Người Được Phép (Logic Service)

Nếu dùng `logic_service.py`, sửa:

```python
ALLOWED_NAMES = ["tan", "tan2", "ten_khac"]  # Thêm tên vào đây
ALLOWED_HOURS_START = 6   # 6h sáng
ALLOWED_HOURS_END = 22    # 10h tối
```

---

## 📡 API Endpoints

### 1. Nhận Diện Khuôn Mặt

#### POST `/recog` - Nhận diện từ base64 image
```bash
curl -X POST http://localhost:8000/recog \
  -F "image=<base64_encoded_image>" \
  -F "w=640" \
  -F "h=480"
```

#### POST `/upload` - Upload file ảnh
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@image.jpg"
```

**Response:**
```json
{
  "name": "tan",
  "probability": 0.95,
  "message": "Nhan dien thanh cong"
}
```

**Lưu ý:** Nếu nhận diện thành công (probability > 0.8), server sẽ tự động gửi lệnh "OPEN" tới ESP32 qua MQTT.

### 2. Test MQTT

#### GET `/test_mqtt?cmd=OPEN`
```bash
curl "http://localhost:8000/test_mqtt?cmd=OPEN"
```

#### POST `/test_mqtt`
```bash
curl -X POST http://localhost:8000/test_mqtt \
  -H "Content-Type: application/json" \
  -d '{"cmd": "OPEN"}'
```

**Response:**
```json
{
  "status": "success",
  "message": "Da gui lenh OPEN toi ESP32",
  "topic": "door/cmd"
}
```

### 3. API Quản Lý

#### GET `/api/stats` - Thống kê nhận diện
```bash
curl http://localhost:8000/api/stats
```

**Response:**
```json
{
  "total": 100,
  "success": 85,
  "fail": 15,
  "door_status": "Đang mở"
}
```

#### GET `/api/logs` - Lấy log hoạt động
```bash
curl http://localhost:8000/api/logs
```

#### DELETE `/api/logs` - Xóa log
```bash
curl -X DELETE http://localhost:8000/api/logs
```

#### GET `/api/persons` - Danh sách người đã train
```bash
curl http://localhost:8000/api/persons
```

#### GET `/api/docs` - Tài liệu API đầy đủ
```bash
curl http://localhost:8000/api/docs
```

### 4. API Quản Lý Dataset

#### POST `/api/save_capture` - Lưu ảnh chụp
```bash
curl -X POST http://localhost:8000/api/save_capture \
  -F "file=@image.jpg" \
  -F "name=nguyen_van_a"
```

#### POST `/api/train_person` - Train model cho người
```bash
curl -X POST http://localhost:8000/api/train_person \
  -H "Content-Type: application/json" \
  -d '{"name": "nguyen_van_a", "skip_align": false, "timeout": 600}'
```

#### POST `/api/align_person` - Align ảnh cho người
```bash
curl -X POST http://localhost:8000/api/align_person \
  -H "Content-Type: application/json" \
  -d '{"name": "nguyen_van_a", "timeout": 600}'
```

---

## 👤 Thêm Người Mới

### Cách 1: Sử Dụng Script Tự Động (Khuyến Nghị)

```bash
cd D:\Face_Mi_AI\MiAI_FaceRecog_3\src
python add_new_person.py --name "nguyen_van_a"
```

Script này sẽ tự động:
1. Align ảnh từ `raw/` sang `processed/`
2. Train lại classifier với người mới
3. Backup model cũ

### Cách 2: Thủ Công (3 Bước)

#### Bước 1: Thu Thập Ảnh

**Yêu cầu:**
- Ít nhất **10-20 ảnh** cho mỗi người (càng nhiều càng tốt)
- Ảnh rõ nét, ánh sáng đủ
- Khuôn mặt nhìn thẳng hoặc góc nhỏ

**Tạo thư mục và copy ảnh:**
```bash
mkdir Dataset\FaceData\raw\nguyen_van_a
# Copy ít nhất 10-20 ảnh vào thư mục này
```

#### Bước 2: Align Ảnh

```bash
cd D:\Face_Mi_AI\MiAI_FaceRecog_3\src
python align_dataset_mtcnn.py \
  --input_dir ../Dataset/FaceData/raw/nguyen_van_a \
  --output_dir ../Dataset/FaceData/processed/nguyen_van_a \
  --image_size 160 \
  --margin 32
```

**Kiểm tra kết quả:**
- Xem thư mục `Dataset/FaceData/processed/nguyen_van_a/`
- Nếu có ảnh = Align thành công!

#### Bước 3: Train Lại Classifier

**Backup model cũ:**
```bash
copy Models\facemodel.pkl Models\facemodel_backup.pkl
```

**Train lại:**
```bash
cd D:\Face_Mi_AI\MiAI_FaceRecog_3\src
python training_optimized.py \
  --facenet_model ../Models/20180402-114759/20180402-114759.pb \
  --data_dir ../Dataset/FaceData/processed \
  --output ../Models/facemodel_new.pkl \
  --use_svm \
  --normalize \
  --test_split 0.2
```

**Restart AI Server:**
```bash
# Dừng server cũ (Ctrl+C) và chạy lại
python face_rec_flask.py
```

### Cách 3: Sử Dụng Web Interface

1. Truy cập: `http://localhost:8000/capture`
2. Chụp ảnh hoặc upload ảnh
3. Nhập tên người
4. Click "Train" để tự động align và train

---

## 📁 Cấu Trúc File

### File Quan Trọng

#### Core Files (Bắt Buộc)
- `src/face_rec_flask.py` ⭐ - Server Flask chính
- `src/main_recognition.py` - Recognition engine
- `src/face_detect.py` - Face detection với MTCNN
- `src/face_align.py` - Face alignment
- `src/face_embedding.py` - Extract embeddings
- `src/face_compare.py` - So sánh embeddings
- `src/facenet.py` - FaceNet utilities
- `src/align/detect_face.py` + `.npy` files - MTCNN models

#### Operational Files
- `src/camera_client.py` - Camera client
- `src/training_optimized.py` - Training classifier
- `src/add_new_person.py` - Thêm người mới tự động
- `src/capture_face_dataset.py` - Thu thập dữ liệu
- `src/align_dataset_mtcnn.py` - Align dataset
- `src/classifier_optimized.py` - Classifier utilities

#### Optional Files
- `src/logic_service.py` - Logic service (tùy chọn)

### Thư Mục Quan Trọng

```
MiAI_FaceRecog_3/
├── src/                    # Source code
│   ├── align/              # MTCNN models
│   └── templates/          # HTML templates
├── Models/                 # AI models
│   ├── 20180402-114759/    # FaceNet model
│   └── facemodel_new.pkl   # Classifier model
└── Dataset/               # Dataset
    └── FaceData/
        ├── raw/            # Ảnh gốc
        └── processed/      # Ảnh đã align
```

---

## 🔍 Kiểm Tra và Debug

### Kiểm Tra AI Server

- Truy cập: `http://localhost:8000/` → Sẽ thấy "OK!" hoặc redirect đến `/capture`
- Xem log trong terminal để thấy kết quả nhận diện

### Kiểm Tra MQTT

- Dùng MQTT Explorer hoặc HiveMQ WebSocket Client
- Kết nối tới broker và subscribe topic `door/cmd`
- Xem có message "OPEN" được gửi không

**Test qua browser:**
```
http://localhost:8000/test_mqtt?cmd=OPEN
http://localhost:8000/test_mqtt?cmd=CLOSE
```

### Kiểm Tra ESP32

- Mở Serial Monitor (115200 baud)
- Xem log kết nối WiFi và MQTT
- Khi nhận lệnh sẽ thấy: "📥 Nhận lệnh: OPEN"

---

## 🐛 Xử Lý Lỗi

### Lỗi: "ModuleNotFoundError: No module named 'tensorflow'"
- Đảm bảo đã cài Python 3.8-3.10
- Chạy: `pip install tensorflow==2.10.0`

### Lỗi: "Không thể kết nối MQTT"
- Kiểm tra internet
- Kiểm tra username/password MQTT trong `face_rec_flask.py`
- Thử tắt firewall tạm thời
- Kiểm tra log trong terminal

### Lỗi: "Không thể mở camera"
- Kiểm tra camera đã kết nối
- Thử đổi `--camera 1` hoặc `--camera 2`
- Nếu dùng RTSP: `python camera_client.py --camera rtsp://192.168.1.100:554/stream`

### ESP32 Không Nhận Lệnh
- Kiểm tra Serial Monitor xem ESP32 đã kết nối MQTT chưa
- Kiểm tra topic: phải là `door/cmd` (không có khoảng trắng)
- Kiểm tra QoS: phải là 1
- Kiểm tra MQTT server có đúng không

### Model Không Load Được
- Kiểm tra file `Models/facemodel_new.pkl` hoặc `Models/facemodel.pkl` có tồn tại không
- Kiểm tra log trong terminal để xem lỗi cụ thể
- Có thể cần retrain model với numpy version hiện tại

### Nhận Diện Không Chính Xác
- Kiểm tra số lượng ảnh train (cần ít nhất 10-20 ảnh/người)
- Kiểm tra chất lượng ảnh (rõ nét, ánh sáng đủ)
- Thử giảm ngưỡng nhận diện (MIN_CONFIDENCE_THRESHOLD)
- Retrain với nhiều ảnh hơn

---

## 📝 Lưu Ý Quan Trọng

### Bảo Mật
- Hiện tại đang dùng `tls_insecure_set(True)` để test
- Khi deploy thực tế nên:
  - Tải CA certificate từ HiveMQ
  - Dùng `client.tls_set(ca_certs="path/to/ca.crt")`
  - Đặt `tls_insecure_set(False)`

### Hiệu Năng
- Giảm `interval` trong camera_client nếu muốn phản hồi nhanh hơn
- Tăng `interval` nếu CPU quá tải
- Mặc định: gửi ảnh mỗi 2 giây

### Logging
- Tất cả log được in ra console
- Có thể redirect vào file:
```bash
python face_rec_flask.py > server.log 2>&1
```

### Backup
- **Luôn backup model cũ** trước khi train
- Nếu train lỗi, có thể khôi phục lại
- Model được backup tự động với timestamp

---

## 🎯 Kết Quả Mong Đợi

Khi hệ thống chạy đúng:

1. ✅ Camera capture ảnh và gửi lên AI server
2. ✅ AI server nhận diện khuôn mặt
3. ✅ Nếu nhận diện thành công (probability > 0.8):
   - AI server gửi lệnh "OPEN" qua MQTT
   - ESP32 nhận lệnh và mở cửa (servo quay 90 độ)
   - Sau 7 giây, ESP32 tự đóng cửa
4. ✅ Log hiển thị trên các terminal và Serial Monitor

---

## 📞 Hỗ Trợ

Nếu gặp vấn đề:
1. Kiểm tra log trong terminal
2. Kiểm tra Serial Monitor ESP32
3. Xem phần [Xử Lý Lỗi](#xử-lý-lỗi) ở trên
4. Kiểm tra API docs: `http://localhost:8000/api/docs`

---

**Chúc bạn sử dụng thành công! 🎉**
