from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from flask import Flask
from flask import render_template , request, jsonify, redirect, url_for
from flask_cors import CORS, cross_origin
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

data_lock = threading.Lock()
activity_logs = deque(maxlen=300)
stats_counters = {"total": 0, "success": 0, "fail": 0}
door_status_text = "Chưa xác định"

def add_log_entry(message, level="info", extra=None):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "level": level,
        "message": message
    }
    if extra:
        entry["extra"] = extra
    with data_lock:
        activity_logs.appendleft(entry)

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



app = Flask(__name__)
CORS(app)
mqtt_client = None


def init_mqtt_client():
    global mqtt_client
    mqtt_client = mqtt.Client(client_id="FaceRecogServer")
    mqtt_client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
    # Co the cung cap file CA tai day: mqtt_client.tls_set(ca_certs="path/to/ca.crt")
    mqtt_client.tls_set(cert_reqs=ssl.CERT_REQUIRED)
    if MQTT_TLS_INSECURE:
        mqtt_client.tls_insecure_set(True)
    try:
        mqtt_client.connect(MQTT_SERVER, MQTT_PORT, keepalive=60)
        mqtt_client.loop_start()
        print("[OK] Da ket noi toi MQTT broker")
        app.logger.info("Da ket noi toi MQTT broker")
    except Exception as exc:
        print(f"[ERROR] Khong the ket noi MQTT: {exc}")
        app.logger.error("Khong the ket noi MQTT: %s", exc)
        mqtt_client = None


def publish_door_command(open_door=True):
    cmd = "OPEN" if open_door else "CLOSE"
    if mqtt_client is None:
        warning = "MQTT client chua duoc ket noi"
        print(f"[WARN] {warning}")
        add_log_entry(f"Không thể gửi lệnh {cmd}: {warning}", level="error")
        return False
    try:
        mqtt_client.publish(MQTT_TOPIC_DOOR_CMD, cmd, qos=1)
        app.logger.info("Da gui lenh %s toi topic %s", cmd, MQTT_TOPIC_DOOR_CMD)
        set_door_status(open_door)
        add_log_entry(f"Đã gửi lệnh {cmd} tới cửa", level="success")
        return True
    except Exception as exc:
        app.logger.error("Gui lenh cua that bai: %s", exc)
        add_log_entry(f"Gửi lệnh {cmd} thất bại: {exc}", level="error")
        return False



@app.route('/')
@cross_origin()
def index():
    return redirect(url_for('capture_panel'))

@app.route('/recog', methods=['POST'])
@cross_origin()
def upload_img_file():
    if model is None:
        return json.dumps({"error": "Model chua duoc load. Vui long kiem tra log."}), 400
    
    if request.method == 'POST':
        # base 64
        name = "Unknown"
        try:
            f = request.form.get('image')
            w = int(request.form.get('w', 640))
            h = int(request.form.get('h', 480))

            decoded_string = base64.b64decode(f)
            frame = np.frombuffer(decoded_string, dtype=np.uint8)
            frame = cv2.imdecode(frame, cv2.IMREAD_ANYCOLOR)

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
                
                # Resize và preprocess
                scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                    interpolation=cv2.INTER_CUBIC)
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
                
                print("Name: {}, Probability: {:.4f}".format(best_name, probability))

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
                    print(f"[OK] Nhan dien thanh cong: {name} (xac suat: {probability:.4f}, threshold: {dynamic_threshold:.2f})")

                    # Chế độ trực tiếp: mở cửa ngay
                    if MODE == "direct":
                        publish_door_command(open_door=True)
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
                    print(f"[UNKNOWN] Khong nhan dien duoc (xac suat: {probability:.4f}, threshold: {dynamic_threshold:.2f})")
        except Exception as e:
            print(f"[ERROR] Loi xu ly anh: {e}")
            return json.dumps({"error": str(e)}), 400

        return name


@app.route('/upload', methods=['POST'])
@cross_origin()
def upload_file():
    """Endpoint để upload file ảnh trực tiếp (không cần camera)"""
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
        
        # Resize và preprocess
        scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
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
            
            # Gửi lệnh mở cửa
            if MODE == "direct":
                publish_door_command(open_door=True)
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


@app.route('/control')
@cross_origin()
def legacy_control_panel():
    return redirect(url_for('capture_panel'))

@app.route('/api/stats', methods=['GET'])
@cross_origin()
def api_stats():
    with data_lock:
        payload = {
            "total": stats_counters["total"],
            "success": stats_counters["success"],
            "fail": stats_counters["fail"],
            "door_status": door_status_text
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

@app.route('/api/persons', methods=['GET'])
@cross_origin()
def api_persons():
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

@app.route('/api/save_capture', methods=['POST'])
@cross_origin()
def api_save_capture():
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
            "processed_count": processed_count
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
    """Endpoint để test MQTT - gửi lệnh mở/đóng cửa thủ công"""
    cmd = request.args.get('cmd', 'OPEN')  # GET: ?cmd=OPEN hoặc ?cmd=CLOSE
    if request.method == 'POST':
        data = request.get_json()
        if data:
            cmd = data.get('cmd', 'OPEN')
    
    if cmd.upper() not in ['OPEN', 'CLOSE']:
        return json.dumps({"error": "Lenh khong hop le. Dung OPEN hoac CLOSE"}), 400
    
    success = publish_door_command(open_door=(cmd.upper() == 'OPEN'))
    
    if success:
        return json.dumps({
            "status": "success",
            "message": f"Da gui lenh {cmd.upper()} toi ESP32",
            "topic": MQTT_TOPIC_DOOR_CMD
        }), 200
    else:
        return json.dumps({
            "status": "error",
            "message": "Khong the ket noi MQTT"
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

