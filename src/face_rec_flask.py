from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import io

# Fix encoding cho Windows console
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except:
        pass

from flask import Flask
from flask import render_template , request, jsonify, redirect, url_for
from flask_cors import CORS, cross_origin
try:
    from flasgger import Swagger
    SWAGGER_AVAILABLE = True
except ImportError:
    SWAGGER_AVAILABLE = False
    print("[WARNING] flasgger not installed. Swagger UI will not be available.")
import tensorflow as tf
import argparse
import facenet
import sys
import math
import pickle
import align.detect_face
import numpy as np
import cv2
from collections import deque
import threading
from sklearn.svm import SVC
import base64
import paho.mqtt.client as mqtt
import ssl
import json
from datetime import datetime
import os
import subprocess
import time
import re

MINSIZE = 20
THRESHOLD = [0.6, 0.7, 0.7]
FACTOR = 0.709
IMAGE_SIZE = 182
INPUT_IMAGE_SIZE = 160
# Tối ưu: Dùng model mới đã train với pipeline nhất quán
CLASSIFIER_PATH = '../Models/facemodel_new.pkl'  # Model mới với normalization
# Fallback về model cũ nếu model mới không có
CLASSIFIER_PATH_OLD = '../Models/facemodel.pkl'
FACENET_MODEL_PATH = '../Models/20180402-114759/20180402-114759.pb'
# Tối ưu: Thêm margin khi crop khuôn mặt (giống như khi align dataset)
FACE_MARGIN = 32  # Margin pixels khi crop (tăng độ chính xác)
# Tối ưu: Threshold động - SVM probability thường thấp hơn cosine similarity
# SVM: 0.25-0.35 là tốt, Cosine: 0.6-0.7 là tốt
MIN_CONFIDENCE_THRESHOLD = 0.30  # Điều chỉnh cho SVM probability (thấp hơn)

# Cau hinh MQTT de giao tiep voi ESP32
MQTT_SERVER = "7a28606d7a234d84a5035fa5e28698a3.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_USER = "nguyenluc0112"
MQTT_PASSWORD = "buithanhTan@123"
MQTT_TOPIC_DOOR_CMD = "door/cmd"
MQTT_TOPIC_AI_RESULT = "ai/result"  # Topic để publish kết quả cho logic service
MQTT_TLS_INSECURE = True  # Dat False neu ban su dung CA chinh xac

# Chế độ hoạt động: "direct" = trực tiếp mở cửa, "publish" = chỉ publish kết quả
MODE = "direct"  # Đổi thành "publish" nếu dùng logic_service.py

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "Dataset", "FaceData", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "Dataset", "FaceData", "processed")

# Cấu hình embeddings database (nếu muốn dùng)
EMBEDDINGS_DB_PATH = os.path.join(BASE_DIR, "Models", "embeddings_database.json")
USE_EMBEDDINGS_DB = os.path.exists(EMBEDDINGS_DB_PATH)  # Tự động bật nếu có database

# Global recognizer (nếu dùng pipeline đồng nhất)
unified_recognizer = None

# Global face capture web (cho quality check khi thu thập ảnh)
face_capture_web = None

data_lock = threading.Lock()
activity_logs = deque(maxlen=300)
stats_counters = {"total": 0, "success": 0, "fail": 0}
door_status_text = "Chưa xác định"

# Lưu log lâu dài vào file
LOG_FILE_DIR = os.path.join(BASE_DIR, "Logs")
os.makedirs(LOG_FILE_DIR, exist_ok=True)
ACCESS_LOG_FILE = os.path.join(LOG_FILE_DIR, "access_history.json")
ACTIVITY_LOG_FILE = os.path.join(LOG_FILE_DIR, "activity_log.json")

# Debounce mechanism để tránh gửi lệnh liên tục
last_door_command_time = {}  # {person_name: timestamp}
DOOR_COMMAND_COOLDOWN = 10  # Giây - thời gian chờ giữa các lệnh cho cùng 1 người

# Lock để tránh reconnect đồng thời
mqtt_reconnect_lock = threading.Lock()
last_reconnect_attempt = 0
RECONNECT_COOLDOWN = 10  # Giây - thời gian chờ giữa các lần reconnect (tăng lên để tránh loop)
mqtt_reconnecting = False  # Flag để tránh reconnect loop
mqtt_last_status = None  # Track trạng thái MQTT để tránh log trùng lặp
last_log_time = {}  # Track thời gian log cuối cùng để debounce
mqtt_initial_connect_logged = False  # Flag để chỉ log kết nối thành công lần đầu khi server khởi động

def add_log_entry(message, level="info", extra=None, debounce_seconds=5):
    """
    Thêm log entry với debounce để tránh log trùng lặp
    
    Args:
        message: Nội dung log
        level: Mức độ log (info, success, warning, error)
        extra: Thông tin bổ sung
        debounce_seconds: Số giây để debounce log giống nhau (0 = không debounce)
    """
    global last_log_time
    
    # Debounce: Kiểm tra nếu log giống nhau trong khoảng thời gian ngắn
    if debounce_seconds > 0:
        current_time = time.time()
        log_key = f"{message}_{level}"
        last_time = last_log_time.get(log_key, 0)
        
        if current_time - last_time < debounce_seconds:
            # Bỏ qua log này vì đã log gần đây
            return
        
        last_log_time[log_key] = current_time
    
    entry = {
        "timestamp": datetime.now().isoformat(),
        "level": level,
        "message": message
    }
    if extra:
        entry["extra"] = extra
    with data_lock:
        activity_logs.appendleft(entry)
    
    # Lưu vào file để lưu lâu dài
    try:
        log_data = {
            "timestamp": entry["timestamp"],
            "level": level,
            "message": message,
            "extra": extra
        }
        _append_to_log_file(ACTIVITY_LOG_FILE, log_data)
    except Exception as e:
        print(f"[WARNING] Khong the luu log vao file: {e}")

def _append_to_log_file(log_file, data):
    """Thêm entry vào file log JSON"""
    try:
        # Đọc file hiện tại
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                try:
                    logs = json.load(f)
                except:
                    logs = []
        else:
            logs = []
        
        # Thêm entry mới
        logs.append(data)
        
        # Giới hạn số lượng log (giữ 10000 entries gần nhất)
        if len(logs) > 10000:
            logs = logs[-10000:]
        
        # Ghi lại file
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[ERROR] Loi khi ghi log file: {e}")

def record_access_event(person_name, confidence, action="enter"):
    """
    Ghi lại sự kiện ra vào nhà
    
    Args:
        person_name: Tên người
        confidence: Độ tin cậy
        action: "enter" hoặc "exit"
    """
    access_entry = {
        "timestamp": datetime.now().isoformat(),
        "person_name": person_name,
        "confidence": float(confidence),
        "action": action,
        "door_command_sent": True
    }
    
    try:
        _append_to_log_file(ACCESS_LOG_FILE, access_entry)
        add_log_entry(
            f"{person_name} {'vào' if action == 'enter' else 'ra'} nhà (confidence: {confidence*100:.2f}%)",
            level="success",
            extra={"person": person_name, "action": action}
        )
    except Exception as e:
        print(f"[ERROR] Loi khi ghi access log: {e}")

def record_stats(success):
    with data_lock:
        stats_counters["total"] += 1
        if success:
            stats_counters["success"] += 1
        else:
            stats_counters["fail"] += 1

def record_recognition_event(name, probability=None, message=""):
    success = name not in [None, "", "Unknown"]
    record_stats(success)
    prob_text = ""
    if probability is not None:
        prob_text = f" ({probability * 100:.2f}%)"
    if success:
        log_msg = f"Nhận diện {name}{prob_text}. {message}".strip()
        add_log_entry(log_msg, level="success")
    else:
        detail = message or "Kết quả không chính xác"
        add_log_entry(f"Nhận diện thất bại{prob_text}. {detail}".strip(), level="error")

def set_door_status(is_open):
    global door_status_text
    with data_lock:
        door_status_text = "Đang mở" if is_open else "Đang đóng"

def sanitize_person_name(name):
    if not name:
        return ""
    name = name.strip().lower()
    name = name.replace(' ', '_')
    name = re.sub(r'[^a-z0-9_]', '', name)
    return name

def crop_face_optimized(frame, bounding_box, margin=32):
    """
    Tối ưu: Crop khuôn mặt với margin và boundary checking
    Đây là hàm QUAN TRỌNG - crop sai sẽ làm nhận diện sai hoàn toàn!
    """
    img_size = np.asarray(frame.shape)[0:2]
    det = np.squeeze(bounding_box[0:4])
    
    # Tính bounding box với margin
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[0] - margin/2, 0)  # x1
    bb[1] = np.maximum(det[1] - margin/2, 0)    # y1
    bb[2] = np.minimum(det[2] + margin/2, img_size[1])  # x2
    bb[3] = np.minimum(det[3] + margin/2, img_size[0])  # y2
    
    # Crop khuôn mặt từ frame
    cropped = frame[int(bb[1]):int(bb[3]), int(bb[0]):int(bb[2]), :]
    
    # Kiểm tra kích thước hợp lệ
    if cropped.size == 0 or cropped.shape[0] < 10 or cropped.shape[1] < 10:
        raise ValueError(f"Khuon mat qua nho sau khi crop: {cropped.shape}")
    
    return cropped, bb

def count_images(folder):
    if not os.path.isdir(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

# Load The Custom Classifier
# Import numpy trước và tạo workaround cho numpy._core nếu cần
import numpy as np

# Workaround cho lỗi numpy._core (pickle file có thể được tạo với numpy 2.0+)
import types
import sys

# Tạo fake numpy._core module trước khi load pickle
# Lưu original pickle.load để không conflict với detect_face.py
_original_pickle_load = pickle.load

if not hasattr(np, '_core'):
    np._core = types.ModuleType('_core')
    # Copy toàn bộ np.core sang np._core, bao gồm cả multiarray
    if hasattr(np, 'core'):
        # Copy module multiarray
        if hasattr(np.core, 'multiarray'):
            np._core.multiarray = np.core.multiarray
        # Copy module umath
        if hasattr(np.core, 'umath'):
            np._core.umath = np.core.umath
        # Copy các thuộc tính khác
        for attr_name in dir(np.core):
            if not attr_name.startswith('_') and attr_name not in ['multiarray', 'umath']:
                try:
                    setattr(np._core, attr_name, getattr(np.core, attr_name))
                except:
                    pass
    # Thêm vào sys.modules để pickle có thể tìm thấy
    sys.modules['numpy._core'] = np._core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray if hasattr(np.core, 'multiarray') else types.ModuleType('multiarray')

# Khôi phục pickle.load sau khi load classifier
pickle.load = _original_pickle_load

# Thử load classifier với nhiều cách
model = None
class_names = None

# Tối ưu: Thử load model mới trước, nếu không có thì dùng model cũ
model_loaded = False
for model_path in [CLASSIFIER_PATH, CLASSIFIER_PATH_OLD]:
    if not os.path.exists(model_path):
        continue
    
    try:
        file = open(model_path, 'rb')
        loaded_data = pickle.load(file)
        file.close()
        
        # Tối ưu: Hỗ trợ cả format cũ (tuple) và format mới (dict với scaler)
        if isinstance(loaded_data, tuple):
            # Format cũ: (model, class_names)
            model, class_names = loaded_data
            scaler = None
        elif isinstance(loaded_data, dict):
            # Format mới: {'model': model, 'class_names': class_names, 'scaler': scaler}
            model = loaded_data['model']
            class_names = loaded_data['class_names']
            scaler = loaded_data.get('scaler', None)
            if scaler is not None:
                print("[OK] Found normalization scaler in model")
        else:
            raise ValueError("Unknown model format")
        
        print(f"[OK] Custom Classifier da load thanh cong: {model_path}")
        model_loaded = True
        break
    except Exception as e:
        print(f"[WARNING] Khong the load {model_path}: {e}")
        continue

if not model_loaded:
    # Nếu không load được cả 2 model, thử với encoding latin1
    print("[WARNING] Khong load duoc model tu ca 2 duong dan, thu voi encoding latin1...")
    try:
        for model_path in [CLASSIFIER_PATH, CLASSIFIER_PATH_OLD]:
            if not os.path.exists(model_path):
                continue
            try:
                with open(model_path, 'rb') as file:
                    loaded_data = pickle.load(file, encoding='latin1')
                    
                    # Hỗ trợ cả format cũ và mới
                    if isinstance(loaded_data, tuple):
                        model, class_names = loaded_data
                        scaler = None
                    elif isinstance(loaded_data, dict):
                        model = loaded_data['model']
                        class_names = loaded_data['class_names']
                        scaler = loaded_data.get('scaler', None)
                    else:
                        raise ValueError("Unknown model format")
                print(f"[OK] Custom Classifier da load thanh cong (voi encoding latin1): {model_path}")
                model_loaded = True
                break
            except Exception as e:
                continue
    except Exception as e2:
        print(f"[ERROR] Van khong the load: {e2}")

if not model_loaded:
    print("[WARNING] Server se chay nhung khong the nhan dien khuon mat")
    print("[INFO] Ban co the test MQTT bang endpoint /test_mqtt")
    # Tạo model giả để server vẫn chạy được
    model = None
    class_names = []
    scaler = None

# Khôi phục pickle.load sau khi load classifier (nếu chưa khôi phục)
if pickle.load != _original_pickle_load:
    pickle.load = _original_pickle_load

# Tắt eager execution để dùng graph mode
tf.compat.v1.disable_eager_execution()

# Tạo graph mới và load model
graph = tf.Graph()
sess = None
images_placeholder = None
embeddings = None
phase_train_placeholder = None
pnet = None
rnet = None
onet = None

with graph.as_default():
    # Cai dat GPU neu co
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

    # Load the model
    print('Loading feature extraction model')
    facenet.load_model(FACENET_MODEL_PATH)

    # Get input and output tensors
    images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
    embedding_size = embeddings.get_shape()[1]
    pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "align")

print("[OK] Model da load thanh cong")

# Khởi tạo unified recognizer nếu dùng embeddings database
if USE_EMBEDDINGS_DB:
    try:
        from main_recognition import FaceRecognizer
        unified_recognizer = FaceRecognizer(
            facenet_model_path=FACENET_MODEL_PATH,
            embeddings_db_path=EMBEDDINGS_DB_PATH,
            classifier_model_path=None,  # Không dùng SVM nếu có embeddings database
            image_size=INPUT_IMAGE_SIZE,
            margin=0,  # Dùng margin=0 để đồng nhất với pipeline
            similarity_threshold=0.5,  # Cosine threshold (0.4-0.6 là tốt)
            method='cosine'
        )
        print("[OK] Unified recognizer initialized với embeddings database")
    except Exception as e:
        print(f"[WARNING] Failed to initialize unified recognizer: {e}")
        import traceback
        traceback.print_exc()
        unified_recognizer = None

# Khởi tạo face capture web (cho quality check)
try:
    from face_capture_web import FaceCaptureWeb
    face_capture_web = FaceCaptureWeb(
        facenet_model_path=FACENET_MODEL_PATH,
        gpu_memory_fraction=0.6
    )
    print("[OK] Face Capture Web initialized với quality check")
except Exception as e:
    print(f"[WARNING] Failed to initialize Face Capture Web: {e}")
    face_capture_web = None

app = Flask(__name__)
CORS(app)

# Swagger configuration
if SWAGGER_AVAILABLE:
    swagger_config = {
        "headers": [],
        "specs": [
            {
                "endpoint": "apispec",
                "route": "/apispec.json",
                "rule_filter": lambda rule: True,
                "model_filter": lambda tag: True,
            }
        ],
        "static_url_path": "/flasgger_static",
        "swagger_ui": True,
        "specs_route": "/swagger"
    }
    swagger_template = {
        "swagger": "2.0",
        "info": {
            "title": "Face Recognition API",
            "description": "API cho hệ thống nhận diện khuôn mặt và điều khiển cửa tự động",
            "version": "1.0.0",
            "contact": {
                "name": "API Support"
            }
        },
        "basePath": "/",
        "schemes": ["http", "https"],
        "tags": [
            {
                "name": "Recognition",
                "description": "Nhận diện khuôn mặt"
            },
            {
                "name": "MQTT",
                "description": "Điều khiển cửa qua MQTT"
            },
            {
                "name": "Management",
                "description": "Quản lý người dùng và dữ liệu"
            },
            {
                "name": "Statistics",
                "description": "Thống kê và logs"
            }
        ]
    }
    swagger = Swagger(app, config=swagger_config, template=swagger_template)

mqtt_client = None


def on_mqtt_connect(client, userdata, flags, rc):
    """Callback khi kết nối MQTT thành công - KHÔNG LOG ở đây để tránh spam"""
    global mqtt_last_status
    try:
        if rc == 0:
            # Chỉ cập nhật trạng thái, không log ở callback
            # Log sẽ được thực hiện trong init_mqtt_client() khi server khởi động
            mqtt_last_status = "connected"
        else:
            error_codes = {
                1: "Connection refused - incorrect protocol version",
                2: "Connection refused - invalid client identifier",
                3: "Connection refused - server unavailable",
                4: "Connection refused - bad username or password",
                5: "Connection refused - not authorised"
            }
            error_msg = error_codes.get(rc, f"Unknown error code: {rc}")
            # Chỉ log lỗi khi thực sự có vấn đề và chưa log trước đó
            if mqtt_last_status != "error":
                print(f"[ERROR] MQTT ket noi that bai: {error_msg} (code: {rc})")
                add_log_entry(f"MQTT ket noi that bai: {error_msg}", level="error", debounce_seconds=30)
                mqtt_last_status = "error"
    except Exception as e:
        print(f"[ERROR] Loi trong on_mqtt_connect: {e}")

def on_mqtt_disconnect(client, userdata, rc):
    """Callback khi mất kết nối MQTT"""
    global mqtt_reconnecting, mqtt_last_status
    try:
        # Giải thích code disconnect
        disconnect_codes = {
            0: "Normal disconnect",
            1: "Unexpected disconnect",
            2: "Disconnect with will message",
            3: "Disconnect due to network error",
            4: "Disconnect due to protocol error",
            5: "Disconnect due to client error",
            6: "Disconnect due to server error",
            7: "Network error / Connection lost"
        }
        code_msg = disconnect_codes.get(rc, f"Unknown code: {rc}")
        
        # Chỉ log khi trạng thái thay đổi từ connected -> disconnected
        if mqtt_last_status == "connected":
            if rc != 0:
                print(f"[WARNING] MQTT client da ngat ket noi: {code_msg} (code: {rc})")
                add_log_entry(f"MQTT ngat ket noi: {code_msg}", level="warning", debounce_seconds=10)
                # Không reconnect ngay trong callback để tránh race condition
                # Reconnect sẽ được xử lý trong publish_door_command hoặc test_mqtt
            else:
                print("[INFO] MQTT client da ngat ket noi (normal)")
                add_log_entry("MQTT client da ngat ket noi", level="info", debounce_seconds=10)
            mqtt_last_status = "disconnected"
        # Khong tu dong reconnect o day de tranh loop
        # Reconnect se duoc xu ly boi publish_door_command hoac test_mqtt endpoint
        mqtt_reconnecting = False  # Reset flag khi disconnect
    except Exception as e:
        print(f"[ERROR] Loi trong on_mqtt_disconnect: {e}")
        mqtt_reconnecting = False

def init_mqtt_client():
    """Khởi tạo MQTT client với lock để tránh init đồng thời"""
    global mqtt_client, mqtt_reconnecting, last_reconnect_attempt
    
    with mqtt_reconnect_lock:
        # Kiểm tra cooldown
        current_time = time.time()
        if current_time - last_reconnect_attempt < RECONNECT_COOLDOWN:
            print(f"[INFO] Bo qua khoi tao MQTT do cooldown ({RECONNECT_COOLDOWN - (current_time - last_reconnect_attempt):.1f}s con lai)")
            return
        
        # Kiểm tra nếu đang reconnect
        if mqtt_reconnecting:
            print("[INFO] Dang trong qua trinh reconnect, bo qua")
            return
        
        # Kiểm tra nếu đã kết nối
        if mqtt_client:
            try:
                if mqtt_client.is_connected():
                    print("[INFO] MQTT client da ket noi roi, khong can khoi tao lai")
                    return
            except:
                pass
        
        mqtt_reconnecting = True
        last_reconnect_attempt = current_time
        
        try:
            # Clean up old client nếu có
            if mqtt_client:
                try:
                    mqtt_client.loop_stop()
                    mqtt_client.disconnect()
                except:
                    pass
            
            # Tạo client ID unique để tránh conflict
            import uuid
            client_id = f"FaceRecogServer_{uuid.uuid4().hex[:8]}"
            mqtt_client = mqtt.Client(client_id=client_id)
            mqtt_client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
            
            # Thiết lập SSL/TLS
            mqtt_client.tls_set(cert_reqs=ssl.CERT_REQUIRED)
            if MQTT_TLS_INSECURE:
                mqtt_client.tls_insecure_set(True)
            
            # Đăng ký callbacks
            mqtt_client.on_connect = on_mqtt_connect
            mqtt_client.on_disconnect = on_mqtt_disconnect
            
            # Kết nối với timeout
            print(f"[INFO] Dang ket noi toi MQTT broker: {MQTT_SERVER}:{MQTT_PORT}")
            try:
                # Tăng keepalive lên 120s để tránh timeout
                # Thêm clean_session=False để giữ session khi reconnect
                mqtt_client.connect(MQTT_SERVER, MQTT_PORT, keepalive=120)
                mqtt_client.loop_start()
                
                # Đợi một chút để kết nối được thiết lập
                time.sleep(5)
                
                # Kiểm tra lại sau khi đợi
                if not mqtt_client.is_connected():
                    print("[WARNING] MQTT chua ket noi sau 5 giay, thu lai...")
                    time.sleep(3)
                
                if mqtt_client.is_connected():
                    # Chỉ log lần đầu khi server khởi động
                    global mqtt_initial_connect_logged
                    if not mqtt_initial_connect_logged:
                        print("[OK] Da ket noi toi MQTT broker")
                        add_log_entry("Da ket noi toi MQTT broker", level="success")
                        mqtt_initial_connect_logged = True
                    mqtt_last_status = "connected"
                    mqtt_reconnecting = False
                else:
                    # Chỉ log khi thực sự không kết nối được và chưa log trước đó
                    if mqtt_last_status != "disconnected":
                        print("[WARNING] MQTT client chua ket noi duoc")
                        add_log_entry("MQTT client chua ket noi duoc", level="warning", debounce_seconds=30)
                        mqtt_last_status = "disconnected"
                    mqtt_reconnecting = False
            except Exception as connect_error:
                try:
                    error_msg = str(connect_error).encode('ascii', 'replace').decode('ascii')
                except:
                    error_msg = repr(connect_error)
                print(f"[ERROR] Khong the ket noi MQTT: {error_msg}")
                add_log_entry(f"Khong the ket noi MQTT: {error_msg}", level="error")
                mqtt_client = None
                mqtt_reconnecting = False
        except Exception as exc:
            try:
                error_msg = str(exc).encode('ascii', 'replace').decode('ascii')
            except:
                error_msg = repr(exc)
            print(f"[ERROR] Loi khi khoi tao MQTT client: {error_msg}")
            add_log_entry(f"Loi khi khoi tao MQTT client: {error_msg}", level="error")
            mqtt_client = None
            mqtt_reconnecting = False


def publish_door_command(open_door=True, person_name=None, force=False):
    """
    Gửi lệnh mở/đóng cửa với debounce mechanism
    
    Args:
        open_door: True để mở, False để đóng
        person_name: Tên người (để debounce theo từng người)
        force: Bỏ qua debounce nếu True
    """
    # Debounce: Kiểm tra thời gian từ lần gửi lệnh trước
    if person_name and not force:
        current_time = time.time()
        last_time = last_door_command_time.get(person_name, 0)
        
        if current_time - last_time < DOOR_COMMAND_COOLDOWN:
            remaining = DOOR_COMMAND_COOLDOWN - (current_time - last_time)
            print(f"[INFO] Bo qua lenh cho {person_name} (cooldown: {remaining:.1f}s)")
            return False
    
    cmd = "OPEN" if open_door else "CLOSE"
    
    # Kiểm tra MQTT client
    if mqtt_client is None:
        warning = "MQTT client chua duoc khoi tao"
        print(f"[WARN] {warning}")
        add_log_entry(f"Khong the gui lenh {cmd}: {warning}", level="error")
        return False
    
    # Kiểm tra trạng thái kết nối
    try:
        is_connected = mqtt_client.is_connected()
    except:
        is_connected = False
    
    if not is_connected:
        warning = "MQTT client chua duoc ket noi"
        print(f"[WARN] {warning}")
        add_log_entry(f"Khong the gui lenh {cmd}: {warning}", level="error")
        
        # Thử reconnect một lần
        print("[INFO] Thu ket noi lai MQTT...")
        init_mqtt_client()
        time.sleep(2)  # Đợi kết nối
        
        # Kiểm tra lại sau reconnect
        try:
            is_connected = mqtt_client.is_connected() if mqtt_client else False
        except:
            is_connected = False
            
        if not is_connected:
            add_log_entry(f"Khong the ket noi lai MQTT sau khi thu reconnect", level="error")
            return False
    
    try:
        result = mqtt_client.publish(MQTT_TOPIC_DOOR_CMD, cmd, qos=1)
        
        # Giải thích publish error codes
        publish_codes = {
            0: "Success",
            1: "Connection refused - incorrect protocol version",
            2: "Connection refused - invalid client identifier", 
            3: "Connection refused - server unavailable",
            4: "Connection refused - bad username or password",
            5: "Connection refused - not authorised"
        }
        
        # result.rc == 0 nghĩa là thành công
        if result.rc == 0:
            # Đợi message được publish thành công
            try:
                result.wait_for_publish(timeout=5)
            except:
                pass  # Bỏ qua nếu timeout
            
            app.logger.info("Da gui lenh %s toi topic %s", cmd, MQTT_TOPIC_DOOR_CMD)
            set_door_status(open_door)
            
            # Cập nhật thời gian gửi lệnh
            if person_name:
                last_door_command_time[person_name] = time.time()
            
            add_log_entry(f"Da gui lenh {cmd} toi cua" + (f" cho {person_name}" if person_name else ""), level="success")
            return True
        else:
            error_code_msg = publish_codes.get(result.rc, f"Unknown code {result.rc}")
            error_msg = f"Loi khi publish: {error_code_msg} (code {result.rc})"
            app.logger.error(error_msg)
            add_log_entry(f"Gui lenh {cmd} that bai: {error_msg}", level="error")
            
            # Nếu là lỗi connection (code 4), thử reconnect
            if result.rc == 4:
                print("[INFO] Phat hien loi ket noi, thu reconnect...")
                init_mqtt_client()
            
            return False
    except Exception as exc:
        try:
            exc_msg = str(exc).encode('ascii', 'replace').decode('ascii')
        except:
            exc_msg = repr(exc)
        app.logger.error("Gui lenh cua that bai: %s", exc_msg)
        add_log_entry(f"Gui lenh {cmd} that bai: {exc_msg}", level="error")
        return False



@app.route('/')
@cross_origin()
def index():
    return redirect('/register')

@app.route('/recog', methods=['POST'])
@cross_origin()
def upload_img_file():
    """
    Nhận diện khuôn mặt từ base64 image
    ---
    tags:
      - Recognition
    parameters:
      - in: formData
        name: image
        type: string
        required: true
        description: Base64 encoded image
      - in: formData
        name: w
        type: integer
        description: Image width
      - in: formData
        name: h
        type: integer
        description: Image height
    responses:
      200:
        description: Kết quả nhận diện
        schema:
          type: object
          properties:
            name:
              type: string
              example: "tan"
            probability:
              type: number
              example: 0.95
            message:
              type: string
              example: "Nhan dien thanh cong"
      400:
        description: Lỗi request
    """
    # Ưu tiên dùng unified recognizer nếu có
    if unified_recognizer is not None:
        try:
            f = request.form.get('image')
            if not f:
                return json.dumps({"error": "Khong co du lieu anh"}), 400
            
            # Optimize: decode base64 và decode image trong một bước
            decoded_string = base64.b64decode(f)
            frame = cv2.imdecode(np.frombuffer(decoded_string, dtype=np.uint8), cv2.IMREAD_COLOR)
            
            if frame is None:
                return json.dumps({"error": "Khong the decode anh"}), 400
            
            # Dùng unified recognizer
            name, confidence, is_match, details = unified_recognizer.recognize(frame, return_details=True)
            
            if is_match and name:
                record_recognition_event(name, confidence, "Nhan dien thanh cong")
                # Tự động mở cửa nếu nhận diện được (chỉ gửi 1 lần với debounce)
                if MODE == "direct":
                    if publish_door_command(open_door=True, person_name=name):
                        record_access_event(name, confidence, action="enter")
                elif MODE == "publish" and mqtt_client:
                    result_data = {
                        'name': name,
                        'confidence': confidence,
                        'timestamp': datetime.now().isoformat()
                    }
                    try:
                        mqtt_client.publish(MQTT_TOPIC_AI_RESULT, json.dumps(result_data), qos=1)
                    except Exception as e:
                        print(f"[ERROR] Loi publish ket qua: {e}")
            else:
                record_recognition_event("Unknown", confidence, details.get('error', ''))
            
            return name if is_match and name else "Unknown"
        except Exception as e:
            print(f"[ERROR] Loi khi nhan dien voi unified recognizer: {e}")
            import traceback
            traceback.print_exc()
            # Fallback về cách cũ
            pass
    
    # Cách cũ: Dùng SVM classifier
    if model is None:
        return json.dumps({"error": "Model chua duoc load. Vui long kiem tra log."}), 400
    
    if request.method == 'POST':
        # base 64
        name = "Unknown"
        try:
            f = request.form.get('image')
            w = int(request.form.get('w', 640))
            h = int(request.form.get('h', 480))

            # Optimize: decode base64 và decode image trong một bước
            decoded_string = base64.b64decode(f)
            frame = cv2.imdecode(np.frombuffer(decoded_string, dtype=np.uint8), cv2.IMREAD_COLOR)

            bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
            faces_found = bounding_boxes.shape[0]

            if faces_found > 0:
                # Tối ưu: Chọn khuôn mặt lớn nhất (thường chính xác hơn)
                det = bounding_boxes[:, 0:4]
                face_sizes = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                largest_face_idx = np.argmax(face_sizes)
                best_face_box = bounding_boxes[largest_face_idx]
                
                # Tối ưu: Crop khuôn mặt ĐÚNG CÁCH với margin
                try:
                    cropped, bb = crop_face_optimized(frame, best_face_box, margin=FACE_MARGIN)
                except Exception as e:
                    print(f"[ERROR] Loi khi crop khuon mat: {e}")
                    return json.dumps({"error": f"Loi crop khuon mat: {str(e)}"}), 400
                
                # Resize và preprocess - optimize: dùng INTER_LINEAR nhanh hon INTER_CUBIC
                scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                    interpolation=cv2.INTER_LINEAR)
                scaled = facenet.prewhiten(scaled)
                scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                
                # Extract embedding
                with graph.as_default():
                    feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                    emb_array = sess.run(embeddings, feed_dict=feed_dict)
                
                # Tối ưu: Normalize embedding nếu có scaler
                if scaler is not None:
                    emb_array = scaler.transform(emb_array)
                
                # Predict
                predictions = model.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[
                    np.arange(len(best_class_indices)), best_class_indices]
                best_name = class_names[best_class_indices[0]]
                probability = float(best_class_probabilities[0])
                
                # Giam logging de tang performance - chi log khi can
                # print("Name: {}, Probability: {:.4f}".format(best_name, probability))

                # Tối ưu: Threshold động cho SVM probability
                # SVM probability thường thấp hơn cosine similarity (0.3-0.4 là tốt)
                dynamic_threshold = MIN_CONFIDENCE_THRESHOLD
                if len(class_names) > 10:
                    # Nhiều classes: threshold thấp hơn
                    dynamic_threshold = max(0.25, MIN_CONFIDENCE_THRESHOLD - 0.05)
                elif len(class_names) < 3:
                    # Ít classes: threshold cao hơn một chút
                    dynamic_threshold = min(0.40, MIN_CONFIDENCE_THRESHOLD + 0.05)
                
                # Đảm bảo threshold không quá thấp
                dynamic_threshold = max(0.25, dynamic_threshold)

                if probability > dynamic_threshold:
                    name = best_name
                    # Giam logging de tang performance
                    # print(f"[OK] Nhan dien thanh cong: {name} (xac suat: {probability:.4f}, threshold: {dynamic_threshold:.2f})")

                    # Chế độ trực tiếp: mở cửa ngay (chỉ gửi 1 lần với debounce)
                    if MODE == "direct":
                        if publish_door_command(open_door=True, person_name=name):
                            record_access_event(name, probability, action="enter")
                    # Chế độ publish: gửi kết quả cho logic service xử lý
                    elif MODE == "publish" and mqtt_client:
                        result_data = {
                            'name': name,
                            'probability': probability,
                            'timestamp': datetime.now().isoformat()
                        }
                    try:
                        mqtt_client.publish(MQTT_TOPIC_AI_RESULT, json.dumps(result_data), qos=1)
                        print(f"[SENT] Da publish ket qua len {MQTT_TOPIC_AI_RESULT}")
                    except Exception as e:
                        print(f"[ERROR] Loi publish ket qua: {e}")
                else:
                    name = "Unknown"
                    # Giam logging de tang performance
                    # print(f"[UNKNOWN] Khong nhan dien duoc (xac suat: {probability:.4f}, threshold: {dynamic_threshold:.2f})")
                    record_recognition_event("Unknown", probability, "Khong vuot qua threshold")
        except Exception as e:
            print(f"[ERROR] Loi xu ly anh: {e}")
            return json.dumps({"error": str(e)}), 400

        return name


@app.route('/upload', methods=['POST'])
@cross_origin()
def upload_file():
    """
    Upload file ảnh để nhận diện
    ---
    tags:
      - Recognition
    consumes:
      - multipart/form-data
    parameters:
      - in: formData
        name: file
        type: file
        required: true
        description: File ảnh để nhận diện
    responses:
      200:
        description: Kết quả nhận diện
        schema:
          type: object
          properties:
            name:
              type: string
              example: "tan"
            probability:
              type: number
              example: 0.95
            message:
              type: string
              example: "Nhan dien thanh cong"
      400:
        description: Lỗi request
    """
    """Endpoint để upload file ảnh trực tiếp (không cần camera) - Đã tối ưu với unified pipeline"""
    # Ưu tiên dùng unified recognizer nếu có
    if unified_recognizer is not None:
        try:
            if 'file' not in request.files:
                return json.dumps({"error": "Khong co file"}), 400
            
            file = request.files['file']
            if file.filename == '':
                return json.dumps({"error": "File rong"}), 400
            
            # Đọc ảnh từ file
            file_bytes = file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                record_recognition_event("Unknown", 0.0, "Khong the doc anh")
                return json.dumps({"name": "Unknown", "message": "Khong the doc anh", "probability": 0.0}), 200
            
            # Dùng unified recognizer
            name, confidence, is_match, details = unified_recognizer.recognize(frame, return_details=True)
            
            result = {
                "name": name if is_match and name else "Unknown",
                "probability": float(confidence),
                "message": details.get('error', 'Recognition complete'),
                "is_match": is_match
            }
            
            if is_match and name:
                record_recognition_event(name, confidence, "Nhan dien thanh cong")
                # Tự động mở cửa nếu nhận diện được (chỉ gửi 1 lần với debounce)
                if MODE == "direct":
                    if publish_door_command(open_door=True, person_name=name):
                        record_access_event(name, confidence, action="enter")
                elif MODE == "publish" and mqtt_client:
                    result_data = {
                        'name': name,
                        'confidence': confidence,
                        'timestamp': datetime.now().isoformat()
                    }
                    try:
                        mqtt_client.publish(MQTT_TOPIC_AI_RESULT, json.dumps(result_data), qos=1)
                    except Exception as e:
                        print(f"[ERROR] Loi publish ket qua: {e}")
            else:
                record_recognition_event("Unknown", confidence, details.get('error', ''))
            
            return json.dumps(result), 200
        except Exception as e:
            print(f"[ERROR] Loi khi nhan dien voi unified recognizer: {e}")
            import traceback
            traceback.print_exc()
            # Fallback về cách cũ
            pass
    
    # Cách cũ: Dùng SVM classifier
    if model is None:
        return json.dumps({"error": "Model chua duoc load"}), 400
    
    if 'file' not in request.files:
        return json.dumps({"error": "Khong co file"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return json.dumps({"error": "File rong"}), 400
    
    try:
        # Đọc ảnh từ file
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return json.dumps({"error": "Khong the doc anh"}), 400
        
        # Xử lý nhận diện (giống như /recog)
        bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
        faces_found = bounding_boxes.shape[0]
        
        if faces_found == 0:
            record_recognition_event("Unknown", 0.0, "Khong tim thay khuon mat")
            return json.dumps({"name": "Unknown", "message": "Khong tim thay khuon mat"}), 200
        
        # Tối ưu: Chọn khuôn mặt lớn nhất
        det = bounding_boxes[:, 0:4]
        face_sizes = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
        largest_face_idx = np.argmax(face_sizes)
        best_face_box = bounding_boxes[largest_face_idx]
        
        # Tối ưu: Crop khuôn mặt ĐÚNG CÁCH với margin
        try:
            cropped, bb = crop_face_optimized(frame, best_face_box, margin=FACE_MARGIN)
        except Exception as e:
            print(f"[ERROR] Loi khi crop khuon mat: {e}")
            record_recognition_event("Unknown", 0.0, f"Loi crop: {str(e)}")
            return json.dumps({"name": "Unknown", "message": f"Loi crop khuon mat: {str(e)}"}), 200
        
        # Resize và preprocess - optimize: dùng INTER_LINEAR nhanh hon INTER_CUBIC
        scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        scaled = facenet.prewhiten(scaled)
        scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
        
        # Extract embedding
        with graph.as_default():
            feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
            emb_array = sess.run(embeddings, feed_dict=feed_dict)
        
        # Tối ưu: Normalize embedding nếu có scaler
        if scaler is not None:
            emb_array = scaler.transform(emb_array)
        
        # Predict
        predictions = model.predict_proba(emb_array)
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
        best_name = class_names[best_class_indices[0]]
        probability = float(best_class_probabilities[0])
        
        # Tối ưu: Threshold động cho SVM probability
        dynamic_threshold = MIN_CONFIDENCE_THRESHOLD
        if len(class_names) > 10:
            # Nhiều classes: threshold thấp hơn
            dynamic_threshold = max(0.25, MIN_CONFIDENCE_THRESHOLD - 0.05)
        elif len(class_names) < 3:
            # Ít classes: threshold cao hơn một chút
            dynamic_threshold = min(0.40, MIN_CONFIDENCE_THRESHOLD + 0.05)
        
        # Đảm bảo threshold không quá thấp
        dynamic_threshold = max(0.25, dynamic_threshold)
        
        result = {
            "name": "Unknown",
            "probability": probability,
            "message": "Khong nhan dien duoc",
            "threshold_used": dynamic_threshold
        }
        
        if probability > dynamic_threshold:
            result["name"] = class_names[best_class_indices[0]]
            result["message"] = "Nhan dien thanh cong"
            print(f"[OK] Nhan dien thanh cong: {result['name']} (xac suat: {result['probability']:.2f})")
            
            # Gửi lệnh mở cửa (chỉ gửi 1 lần với debounce)
            if MODE == "direct":
                if publish_door_command(open_door=True, person_name=result['name']):
                    record_access_event(result['name'], result['probability'], action="enter")
            elif MODE == "publish" and mqtt_client:
                result_data = {
                    'name': result['name'],
                    'probability': result['probability'],
                    'timestamp': datetime.now().isoformat()
                }
                mqtt_client.publish(MQTT_TOPIC_AI_RESULT, json.dumps(result_data), qos=1)
        
        record_recognition_event(result["name"], result.get("probability"), result.get("message"))
        return json.dumps(result), 200
            
    except Exception as e:
        print(f"[ERROR] Loi xu ly file: {e}")
        add_log_entry(f"Loi xu ly file upload: {e}", level="error")
        return json.dumps({"error": str(e)}), 400


def render_mode_page(mode_name):
    try:
        return render_template('control_panel.html', page_mode=mode_name)
    except Exception as e:
        return f"Error loading control panel ({mode_name}): {e}", 500


# New routes for refactored UI
@app.route('/register')
@cross_origin()
def register_page():
    return render_template('register.html', active_page='register')

@app.route('/scan')
@cross_origin()
def scan_page():
    return render_template('scan.html', active_page='scan')

@app.route('/admin')
@cross_origin()
def admin_page():
    return render_template('admin.html', active_page='admin')

# Legacy routes (keep for backward compatibility)
@app.route('/capture')
@cross_origin()
def capture_panel():
    return render_mode_page('capture')


@app.route('/auto-door')
@cross_origin()
def auto_door_panel():
    return render_mode_page('auto')


@app.route('/manual-control')
@cross_origin()
def manual_control_panel():
    return render_mode_page('manual')

@app.route('/access-history')
@cross_origin()
def access_history_panel():
    return render_mode_page('history')

@app.route('/control')
@cross_origin()
def legacy_control_panel():
    return redirect(url_for('capture_panel'))

@app.route('/api/stats', methods=['GET'])
@cross_origin()
def api_stats():
    """
    Lấy thống kê hệ thống
    ---
    tags:
      - Statistics
    responses:
      200:
        description: Thống kê hệ thống
        schema:
          type: object
          properties:
            total:
              type: integer
              example: 100
            success:
              type: integer
              example: 85
            fail:
              type: integer
              example: 15
            door_status:
              type: string
              example: "Đóng"
            mqtt_connected:
              type: boolean
              example: true
    """
    # Kiểm tra trạng thái MQTT
    mqtt_connected = False
    if mqtt_client:
        try:
            mqtt_connected = mqtt_client.is_connected()
        except:
            mqtt_connected = False
    
    with data_lock:
        payload = {
            "total": stats_counters["total"],
            "success": stats_counters["success"],
            "fail": stats_counters["fail"],
            "door_status": door_status_text,
            "mqtt_connected": mqtt_connected
        }
    return jsonify(payload)

@app.route('/api/logs', methods=['GET', 'DELETE'])
@cross_origin()
def api_logs():
    if request.method == 'DELETE':
        with data_lock:
            activity_logs.clear()
        add_log_entry("Đã xóa toàn bộ log từ giao diện", level="info")
        return jsonify({"status": "cleared"})
    with data_lock:
        logs = list(activity_logs)
    return jsonify({"logs": logs})

@app.route('/api/access_history', methods=['GET'])
@cross_origin()
def api_access_history():
    """API để xem lịch sử ra vào nhà"""
    try:
        limit = int(request.args.get('limit', 100))  # Mặc định 100 entries gần nhất
        person_name = request.args.get('person', None)  # Lọc theo người
        
        if os.path.exists(ACCESS_LOG_FILE):
            with open(ACCESS_LOG_FILE, 'r', encoding='utf-8') as f:
                try:
                    history = json.load(f)
                except:
                    history = []
        else:
            history = []
        
        # Lọc theo người nếu có
        if person_name:
            history = [h for h in history if h.get('person_name', '').lower() == person_name.lower()]
        
        # Sắp xếp theo thời gian (mới nhất trước)
        history.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        # Giới hạn số lượng
        history = history[:limit]
        
        return jsonify({
            "success": True,
            "total": len(history),
            "history": history
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/access_stats', methods=['GET'])
@cross_origin()
def api_access_stats():
    """API để xem thống kê ra vào"""
    try:
        if os.path.exists(ACCESS_LOG_FILE):
            with open(ACCESS_LOG_FILE, 'r', encoding='utf-8') as f:
                try:
                    history = json.load(f)
                except:
                    history = []
        else:
            history = []
        
        # Thống kê theo người
        person_stats = {}
        for entry in history:
            person = entry.get('person_name', 'Unknown')
            if person not in person_stats:
                person_stats[person] = {
                    "name": person,
                    "total_entries": 0,
                    "last_access": None
                }
            person_stats[person]["total_entries"] += 1
            timestamp = entry.get('timestamp', '')
            if not person_stats[person]["last_access"] or timestamp > person_stats[person]["last_access"]:
                person_stats[person]["last_access"] = timestamp
        
        # Sắp xếp theo số lần vào
        stats_list = sorted(person_stats.values(), key=lambda x: x["total_entries"], reverse=True)
        
        return jsonify({
            "success": True,
            "total_people": len(person_stats),
            "total_entries": len(history),
            "stats": stats_list
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/persons', methods=['GET'])
@cross_origin()
def api_persons():
    """
    Lấy danh sách người dùng
    ---
    tags:
      - Management
    responses:
      200:
        description: Danh sách người dùng
        schema:
          type: object
          properties:
            persons:
              type: array
              items:
                type: object
                properties:
                  name:
                    type: string
                    example: "tan"
                  raw_count:
                    type: integer
                    example: 100
                  processed_count:
                    type: integer
                    example: 95
    """
    persons = []
    if os.path.isdir(RAW_DATA_DIR):
        for name in sorted(os.listdir(RAW_DATA_DIR)):
            person_raw = os.path.join(RAW_DATA_DIR, name)
            if not os.path.isdir(person_raw):
                continue
            processed_dir = os.path.join(PROCESSED_DATA_DIR, name)
            raw_count = count_images(person_raw)
            processed_count = count_images(processed_dir)
            last_updated = None
            try:
                last_updated = datetime.fromtimestamp(os.path.getmtime(person_raw)).isoformat()
            except:
                last_updated = ""
            persons.append({
                "name": name,
                "raw_count": raw_count,
                "processed_count": processed_count,
                "last_updated": last_updated
            })
    return jsonify({"persons": persons})

@app.route('/api/persons/<name>', methods=['DELETE'])
@cross_origin()
def api_delete_person(name):
    """
    Xóa người dùng khỏi dataset
    ---
    tags:
      - Management
    parameters:
      - in: path
        name: name
        type: string
        required: true
        description: Tên người dùng cần xóa
      - in: query
        name: retrain
        type: boolean
        default: false
        description: Có retrain model sau khi xóa không
    responses:
      200:
        description: Xóa thành công
        schema:
          type: object
          properties:
            status:
              type: string
              example: "success"
            message:
              type: string
              example: "Da xoa nguoi dung 'tan' thanh cong"
            deleted_folders:
              type: array
              items:
                type: string
              example: ["raw/tan", "processed/tan"]
      404:
        description: Không tìm thấy người dùng
      500:
        description: Lỗi server
    """
    import shutil
    
    try:
        # Sanitize tên người dùng
        sanitized_name = sanitize_person_name(name)
        if not sanitized_name:
            return jsonify({
                "status": "error",
                "message": "Ten nguoi dung khong hop le"
            }), 400
        
        # Đường dẫn thư mục cần xóa
        raw_person_dir = os.path.join(RAW_DATA_DIR, sanitized_name)
        processed_person_dir = os.path.join(PROCESSED_DATA_DIR, sanitized_name)
        
        deleted_folders = []
        deleted_files_count = 0
        
        # Xóa thư mục raw
        if os.path.exists(raw_person_dir):
            try:
                # Đếm số file trước khi xóa
                raw_files = []
                for root, dirs, files in os.walk(raw_person_dir):
                    raw_files.extend(files)
                deleted_files_count += len(raw_files)
                
                shutil.rmtree(raw_person_dir)
                deleted_folders.append(f"raw/{sanitized_name}")
                print(f"[INFO] Da xoa thu muc raw: {raw_person_dir}")
            except Exception as e:
                print(f"[ERROR] Khong the xoa thu muc raw: {e}")
                return jsonify({
                    "status": "error",
                    "message": f"Khong the xoa thu muc raw: {str(e)}"
                }), 500
        
        # Xóa thư mục processed
        if os.path.exists(processed_person_dir):
            try:
                # Đếm số file trước khi xóa
                processed_files = []
                for root, dirs, files in os.walk(processed_person_dir):
                    processed_files.extend(files)
                deleted_files_count += len(processed_files)
                
                shutil.rmtree(processed_person_dir)
                deleted_folders.append(f"processed/{sanitized_name}")
                print(f"[INFO] Da xoa thu muc processed: {processed_person_dir}")
            except Exception as e:
                print(f"[ERROR] Khong the xoa thu muc processed: {e}")
                return jsonify({
                    "status": "error",
                    "message": f"Khong the xoa thu muc processed: {str(e)}"
                }), 500
        
        # Kiểm tra xem có xóa được gì không
        if not deleted_folders:
            return jsonify({
                "status": "error",
                "message": f"Khong tim thay nguoi dung '{sanitized_name}' trong dataset"
            }), 404
        
        # Log hoạt động
        add_log_entry(
            f"Da xoa nguoi dung '{sanitized_name}' ({deleted_files_count} files)",
            level="info"
        )
        
        # Retrain model nếu được yêu cầu
        retrain = request.args.get('retrain', 'false').lower() == 'true'
        retrain_message = ""
        if retrain:
            try:
                print(f"[INFO] Bat dau retrain model sau khi xoa '{sanitized_name}'...")
                # Import training function
                from training_optimized import FaceTrainer
                trainer = FaceTrainer()
                success = trainer.train_classifier(use_embeddings_db=False)
                if success:
                    retrain_message = " Da retrain model thanh cong."
                    add_log_entry(f"Da retrain model sau khi xoa '{sanitized_name}'", level="success")
                else:
                    retrain_message = " Retrain model that bai."
                    add_log_entry(f"Retrain model that bai sau khi xoa '{sanitized_name}'", level="error")
            except Exception as e:
                retrain_message = f" Loi khi retrain: {str(e)}"
                print(f"[ERROR] Loi khi retrain: {e}")
                add_log_entry(f"Loi khi retrain model: {str(e)}", level="error")
        
        return jsonify({
            "status": "success",
            "message": f"Da xoa nguoi dung '{sanitized_name}' thanh cong.{retrain_message}",
            "deleted_folders": deleted_folders,
            "deleted_files_count": deleted_files_count,
            "retrained": retrain
        }), 200
        
    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] Loi khi xoa nguoi dung: {error_msg}")
        add_log_entry(f"Loi khi xoa nguoi dung '{name}': {error_msg}", level="error")
        return jsonify({
            "status": "error",
            "message": f"Loi khi xoa nguoi dung: {error_msg}"
        }), 500

@app.route('/api/save_capture', methods=['POST'])
@cross_origin()
def api_save_capture():
    """
    Lưu ảnh vào dataset
    ---
    tags:
      - Management
    consumes:
      - multipart/form-data
    parameters:
      - in: formData
        name: name
        type: string
        required: true
        description: Tên người dùng
      - in: formData
        name: file
        type: file
        required: true
        description: File ảnh
    responses:
      200:
        description: Kết quả lưu ảnh
        schema:
          type: object
          properties:
            status:
              type: string
              example: "success"
            person:
              type: string
              example: "tan"
            raw_count:
              type: integer
              example: 101
    """
    """API lưu ảnh với quality check tự động - Đã tối ưu"""
    # Ưu tiên dùng FaceCaptureWeb nếu có (có quality check)
    if face_capture_web is not None:
        try:
            # Nhận base64 hoặc file
            base64_image = request.form.get('image') or request.form.get('base64_image')
            file = request.files.get('file')
            
            name = sanitize_person_name(request.form.get('name', ''))
            if not name:
                return jsonify({"error": "Vui long nhap ten hop le (chu thuong, khong dau)"}), 400
            
            # Convert file sang base64 nếu cần
            if file and not base64_image:
                file_bytes = file.read()
                import base64 as b64
                base64_image = b64.b64encode(file_bytes).decode('utf-8')
            
            if not base64_image:
                return jsonify({"error": "Khong co du lieu anh (can base64 hoac file)"}), 400
            
            # Dùng FaceCaptureWeb để capture và lưu với quality check
            result = face_capture_web.capture_and_save(
                base64_image=base64_image,
                person_name=name,
                raw_data_dir=RAW_DATA_DIR,
                min_interval=0.5
            )
            
            if result['success'] and result.get('filepath'):
                raw_count = count_images(os.path.join(RAW_DATA_DIR, name))
                add_log_entry(f"Đã lưu ảnh chất lượng cao cho '{name}' (quality: {result['quality_score']:.2f})", level="success")
                return jsonify({
                    "status": "success",
                    "message": result['message'],
                    "filename": os.path.basename(result['filepath']),
                    "person": name,
                    "raw_count": raw_count,
                    "quality_score": result['quality_score'],
                    "stats": result.get('stats', {})
                })
            else:
                add_log_entry(f"Bỏ qua ảnh cho '{name}': {result['message']}", level="warning")
                return jsonify({
                    "status": "skipped",
                    "message": result['message'],
                    "quality_score": result.get('quality_score', 0.0),
                    "stats": result.get('stats', {})
                }), 200
        except Exception as e:
            print(f"[ERROR] Loi khi save capture voi FaceCaptureWeb: {e}")
            import traceback
            traceback.print_exc()
            # Fallback về cách cũ
            pass
    
    # Cách cũ: Lưu trực tiếp không quality check
    if 'file' not in request.files:
        return jsonify({"error": "Khong tim thay file upload"}), 400
    name = sanitize_person_name(request.form.get('name', ''))
    if not name:
        return jsonify({"error": "Vui long nhap ten hop le (chu thuong, khong dau)"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "File rong"}), 400
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    person_dir = os.path.join(RAW_DATA_DIR, name)
    os.makedirs(person_dir, exist_ok=True)
    filename = f"{int(time.time()*1000)}.jpg"
    filepath = os.path.join(person_dir, filename)
    try:
        file.save(filepath)
    except Exception as e:
        return jsonify({"error": f"Khong the luu file: {e}"}), 500
    raw_count = count_images(person_dir)
    add_log_entry(f"Đã lưu ảnh cho '{name}' (không quality check)", level="info")
    return jsonify({
        "status": "success",
        "message": f"Da luu anh vao {person_dir}",
        "filename": filename,
        "person": name,
        "raw_count": raw_count
    })

@app.route('/api/train_person', methods=['POST'])
@cross_origin()
def api_train_person():
    data = request.get_json(force=True, silent=True) or {}
    name = sanitize_person_name(data.get('name', ''))
    if not name:
        return jsonify({"error": "Vui long nhap ten hop le"}), 400
    person_dir = os.path.join(RAW_DATA_DIR, name)
    if not os.path.isdir(person_dir) or count_images(person_dir) == 0:
        return jsonify({"error": f"Khong tim thay anh trong {person_dir}. Vui long luu anh truoc."}), 400
    
    # Kiểm tra xem có dùng embeddings database không
    use_embeddings_db = data.get('use_embeddings_db', False)
    embeddings_db_path = data.get('embeddings_db_path', EMBEDDINGS_DB_PATH)
    
    if use_embeddings_db and os.path.exists(embeddings_db_path):
        # Dùng embeddings database - train trực tiếp không cần align
        try:
            from training_optimized import FaceTrainer
            trainer = FaceTrainer(
                facenet_model_path=FACENET_MODEL_PATH,
                data_dir=RAW_DATA_DIR,  # Dùng raw data
                output_model_path=embeddings_db_path,
                image_size=INPUT_IMAGE_SIZE,
                margin=0
            )
            success, message, stats = trainer.train_person_with_embeddings_db(
                person_name=name,
                raw_data_dir=RAW_DATA_DIR,
                db_path=embeddings_db_path
            )
            trainer.close()
            
            # Refresh unified recognizer nếu có
            if unified_recognizer:
                unified_recognizer.refresh_database()
            
            if success:
                add_log_entry(f"Train '{name}' với embeddings database thành công: {message}", level="success")
                return jsonify({
                    "success": True,
                    "message": message,
                    "stats": stats,
                    "method": "embeddings_db"
                })
            else:
                return jsonify({"error": message, "stats": stats}), 500
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": f"Lỗi khi train với embeddings database: {str(e)}"}), 500
    else:
        # Dùng SVM classifier (cách cũ)
        script_path = os.path.join(SRC_DIR, 'add_new_person.py')
        if not os.path.isfile(script_path):
            return jsonify({"error": "Khong tim thay script add_new_person.py"}), 500
        processed_dir = os.path.join(PROCESSED_DATA_DIR, name)
        processed_count = count_images(processed_dir)
        skip_align = False
        if processed_count > 0 or data.get('skip_align'):
            skip_align = True
        cmd = ["py", script_path, "--name", name]
        if skip_align:
            cmd.append("--skip_align")
        try:
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=SRC_DIR,
                timeout=data.get('timeout', 600)
            )
            success = process.returncode == 0
            response = {
                "success": success,
                "cmd": " ".join(cmd),
                "stdout": process.stdout[-5000:],
                "stderr": process.stderr[-5000:],
                "skip_align": skip_align,
                "processed_count": processed_count,
                "method": "svm"
            }
            if success:
                response["message"] = "Train thanh cong. Vui long restart server neu can."
                return jsonify(response)
            else:
                response["message"] = "Train that bai. Vui long xem stderr."
                return jsonify(response), 500
        except subprocess.TimeoutExpired:
            return jsonify({"error": "Qua thoi gian train (timeout)."}), 500
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route('/api/align_person', methods=['POST'])
@cross_origin()
def api_align_person():
    data = request.get_json(force=True, silent=True) or {}
    name = sanitize_person_name(data.get('name', ''))
    if not name:
        return jsonify({"error": "Vui long nhap ten hop le"}), 400
    person_dir = os.path.join(RAW_DATA_DIR, name)
    if not os.path.isdir(person_dir) or count_images(person_dir) == 0:
        return jsonify({"error": f"Khong tim thay anh trong {person_dir}. Vui long luu anh truoc."}), 400
    script_path = os.path.join(SRC_DIR, 'add_new_person.py')
    if not os.path.isfile(script_path):
        return jsonify({"error": "Khong tim thay script add_new_person.py"}), 500
    cmd = ["py", script_path, "--name", name, "--skip_train"]
    try:
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=SRC_DIR,
            timeout=data.get('timeout', 600)
        )
        success = process.returncode == 0
        response = {
            "success": success,
            "cmd": " ".join(cmd),
            "stdout": process.stdout[-5000:],
            "stderr": process.stderr[-5000:],
        }
        if success:
            response["message"] = "Align thanh cong. Ban co the train ngay bay gio."
            return jsonify(response)
        else:
            response["message"] = "Align that bai. Vui long xem stderr."
            return jsonify(response), 500
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Qua thoi gian align (timeout)."}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/test_mqtt', methods=['GET', 'POST'])
@cross_origin()
def test_mqtt():
    """
    Test MQTT - Gửi lệnh mở/đóng cửa thủ công
    ---
    tags:
      - MQTT
    parameters:
      - in: query
        name: cmd
        type: string
        enum: [OPEN, CLOSE]
        description: Lệnh điều khiển cửa
      - in: body
        name: body
        schema:
          type: object
          properties:
            cmd:
              type: string
              enum: [OPEN, CLOSE]
              example: "OPEN"
    responses:
      200:
        description: Kết quả gửi lệnh
        schema:
          type: object
          properties:
            status:
              type: string
              example: "success"
            message:
              type: string
              example: "Da gui lenh OPEN toi ESP32"
            topic:
              type: string
              example: "door/cmd"
      400:
        description: Lỗi request
    """
    try:
        cmd = request.args.get('cmd', 'OPEN')  # GET: ?cmd=OPEN hoặc ?cmd=CLOSE
        if request.method == 'POST':
            data = request.get_json()
            if data:
                cmd = data.get('cmd', 'OPEN')
        
        cmd_upper = cmd.upper()
        if cmd_upper not in ['OPEN', 'CLOSE']:
            return jsonify({
                "status": "error",
                "message": "Lenh khong hop le. Dung OPEN hoac CLOSE"
            }), 400
        
        # Nếu MQTT client chưa được khởi tạo, thử khởi tạo lại
        global mqtt_client
        if mqtt_client is None:
            print("[INFO] MQTT client chua duoc khoi tao, dang thu khoi tao lai...")
            init_mqtt_client()
            time.sleep(2)  # Đợi một chút để kết nối
        
        # Kiểm tra lại sau khi init
        if mqtt_client is None:
            return jsonify({
                "status": "error",
                "message": "Khong the khoi tao MQTT client"
            }), 500
        
        # Kiểm tra kết nối trước khi gửi lệnh
        try:
            is_connected = mqtt_client.is_connected()
        except:
            is_connected = False
        
        if not is_connected:
            print("[WARN] MQTT chua ket noi, dang thu khoi tao lai...")
            # Thử khởi tạo lại client thay vì reconnect để tránh loop
            init_mqtt_client()
            time.sleep(2)
            try:
                is_connected = mqtt_client.is_connected() if mqtt_client else False
            except:
                is_connected = False
        
        # Gửi lệnh với force=True để bỏ qua debounce khi test thủ công
        success = publish_door_command(open_door=(cmd_upper == 'OPEN'), force=True)
        
        if success:
            return jsonify({
                "status": "success",
                "message": f"Da gui lenh {cmd_upper} toi ESP32",
                "topic": MQTT_TOPIC_DOOR_CMD
            }), 200
        else:
            # Kiểm tra lại trạng thái MQTT để đưa ra thông báo chi tiết hơn
            error_detail = "Khong the ket noi MQTT"
            if mqtt_client is None:
                error_detail = "MQTT client chua duoc khoi tao"
            else:
                try:
                    if not mqtt_client.is_connected():
                        error_detail = "MQTT client chua ket noi duoc"
                except:
                    error_detail = "MQTT client co loi"
            
            return jsonify({
                "status": "error",
                "message": error_detail
            }), 500
    except Exception as e:
        try:
            error_msg = str(e).encode('ascii', 'replace').decode('ascii')
        except:
            error_msg = repr(e)
        return jsonify({
            "status": "error",
            "message": f"Loi khi xu ly: {error_msg}"
        }), 500

@app.route('/api/docs', methods=['GET'])
@cross_origin()
def api_docs():
    """API endpoint trả về hướng dẫn sử dụng dưới dạng JSON"""
    docs = {
        "title": "AI Face Recognition Server API Documentation",
        "version": "1.0",
        "base_url": "http://localhost:8000",
        "endpoints": {
            "root": {
                "method": "GET",
                "url": "/",
                "description": "Trang chủ - redirect đến /capture",
                "response": "HTML page"
            },
            "capture_panel": {
                "method": "GET",
                "url": "/capture",
                "description": "Panel chụp ảnh và thêm người mới",
                "response": "HTML page"
            },
            "auto_door_panel": {
                "method": "GET",
                "url": "/auto-door",
                "description": "Panel tự động mở cửa",
                "response": "HTML page"
            },
            "manual_control_panel": {
                "method": "GET",
                "url": "/manual-control",
                "description": "Panel điều khiển thủ công",
                "response": "HTML page"
            },
            "recognize_base64": {
                "method": "POST",
                "url": "/recog",
                "description": "Nhận diện khuôn mặt từ ảnh base64 (từ camera)",
                "content_type": "application/x-www-form-urlencoded",
                "parameters": {
                    "image": "Base64 encoded image (required)",
                    "w": "Width (default: 640)",
                    "h": "Height (default: 480)"
                },
                "response_success": {
                    "name": "string - Tên người được nhận diện hoặc 'Unknown'"
                },
                "response_error": {
                    "error": "string - Mô tả lỗi"
                }
            },
            "upload_image": {
                "method": "POST",
                "url": "/upload",
                "description": "Upload ảnh file để nhận diện (không cần camera)",
                "content_type": "multipart/form-data",
                "parameters": {
                    "file": "Image file (jpg, png, etc.)"
                },
                "response_success": {
                    "name": "string - Tên người được nhận diện",
                    "probability": "float - Xác suất nhận diện (0-1)",
                    "message": "string - Thông báo kết quả"
                },
                "response_error": {
                    "error": "string - Mô tả lỗi"
                },
                "note": "Nếu nhận diện thành công (probability > 0.8), server sẽ tự động gửi lệnh OPEN tới ESP32 qua MQTT"
            },
            "test_mqtt": {
                "method": "GET, POST",
                "url": "/test_mqtt",
                "description": "Test kết nối MQTT - gửi lệnh mở/đóng cửa thủ công",
                "parameters": {
                    "cmd": "OPEN hoặc CLOSE (GET: query param, POST: JSON body)"
                },
                "response_success": {
                    "status": "success",
                    "message": "Da gui lenh {cmd} toi ESP32",
                    "topic": "door/cmd"
                },
                "response_error": {
                    "status": "error",
                    "message": "Khong the ket noi MQTT"
                },
                "examples": {
                    "get": "GET /test_mqtt?cmd=OPEN",
                    "post": "POST /test_mqtt với body: {\"cmd\": \"OPEN\"}"
                }
            },
            "api_stats": {
                "method": "GET",
                "url": "/api/stats",
                "description": "Lấy thống kê nhận diện",
                "response": {
                    "total": "int - Tổng số lần nhận diện",
                    "success": "int - Số lần thành công",
                    "fail": "int - Số lần thất bại",
                    "door_status": "string - Trạng thái cửa"
                }
            },
            "api_logs": {
                "method": "GET, DELETE",
                "url": "/api/logs",
                "description": "Lấy hoặc xóa log hoạt động",
                "get_response": {
                    "logs": "array - Danh sách log entries"
                },
                "delete_response": {
                    "status": "cleared"
                }
            },
            "api_persons": {
                "method": "GET",
                "url": "/api/persons",
                "description": "Lấy danh sách người đã được train",
                "response": {
                    "persons": [
                        {
                            "name": "string - Tên người",
                            "raw_count": "int - Số ảnh raw",
                            "processed_count": "int - Số ảnh đã xử lý",
                            "last_updated": "string - ISO timestamp"
                        }
                    ]
                }
            },
            "api_save_capture": {
                "method": "POST",
                "url": "/api/save_capture",
                "description": "Lưu ảnh chụp được vào dataset",
                "content_type": "multipart/form-data",
                "parameters": {
                    "file": "Image file",
                    "name": "Tên người (chữ thường, không dấu)"
                },
                "response": {
                    "status": "success",
                    "message": "Da luu anh vao {path}",
                    "filename": "string",
                    "person": "string",
                    "raw_count": "int"
                }
            },
            "api_train_person": {
                "method": "POST",
                "url": "/api/train_person",
                "description": "Train model cho một người",
                "content_type": "application/json",
                "parameters": {
                    "name": "Tên người cần train",
                    "skip_align": "boolean - Bỏ qua bước align nếu đã có",
                    "timeout": "int - Timeout tính bằng giây (default: 600)"
                },
                "response": {
                    "success": "boolean",
                    "message": "string",
                    "cmd": "string - Command đã chạy",
                    "stdout": "string - Output",
                    "stderr": "string - Error output"
                }
            },
            "api_align_person": {
                "method": "POST",
                "url": "/api/align_person",
                "description": "Align ảnh cho một người (không train)",
                "content_type": "application/json",
                "parameters": {
                    "name": "Tên người cần align",
                    "timeout": "int - Timeout tính bằng giây (default: 600)"
                },
                "response": {
                    "success": "boolean",
                    "message": "string",
                    "cmd": "string",
                    "stdout": "string",
                    "stderr": "string"
                }
            },
            "api_docs": {
                "method": "GET",
                "url": "/api/docs",
                "description": "API documentation (endpoint này)",
                "response": "JSON - Thông tin tất cả endpoints"
            }
        },
        "mqtt_config": {
            "server": MQTT_SERVER,
            "port": MQTT_PORT,
            "topics": {
                "door_cmd": MQTT_TOPIC_DOOR_CMD,
                "ai_result": MQTT_TOPIC_AI_RESULT
            },
            "mode": MODE,
            "mode_description": "direct: Trực tiếp mở cửa, publish: Chỉ publish kết quả"
        },
        "examples": {
            "test_mqtt_get": "curl 'http://localhost:8000/test_mqtt?cmd=OPEN'",
            "test_mqtt_post": "curl -X POST http://localhost:8000/test_mqtt -H 'Content-Type: application/json' -d '{\"cmd\": \"OPEN\"}'",
            "upload_image": "curl -X POST http://localhost:8000/upload -F 'file=@image.jpg'",
            "recognize_base64": "curl -X POST http://localhost:8000/recog -F 'image=<base64>' -F 'w=640' -F 'h=480'",
            "get_stats": "curl http://localhost:8000/api/stats",
            "get_logs": "curl http://localhost:8000/api/logs",
            "get_persons": "curl http://localhost:8000/api/persons"
        },
        "python_example": {
            "test_mqtt": "import requests\nresponse = requests.get('http://localhost:8000/test_mqtt?cmd=OPEN')\nprint(response.json())",
            "upload_image": "import requests\nwith open('image.jpg', 'rb') as f:\n    files = {'file': f}\n    response = requests.post('http://localhost:8000/upload', files=files)\n    print(response.json())"
        },
        "notes": [
            "Nếu model không load được, server vẫn chạy nhưng không thể nhận diện",
            "Có thể test MQTT bằng endpoint /test_mqtt ngay cả khi model chưa load",
            "Nếu nhận diện thành công (probability > 0.8), server sẽ tự động gửi lệnh OPEN tới ESP32 (nếu MODE = 'direct')",
            "Tất cả endpoints đều hỗ trợ CORS"
        ]
    }
    return jsonify(docs)


if __name__ == '__main__':
    print("[START] Dang khoi dong AI Server...")
    print(f"[INFO] Che do: {MODE} ({'Truc tiep mo cua' if MODE == 'direct' else 'Publish ket qua'})")
    init_mqtt_client()
    print("[INFO] Server dang chay tai: http://0.0.0.0:8000")
    app.run(debug=True, host='0.0.0.0',port='8000')

