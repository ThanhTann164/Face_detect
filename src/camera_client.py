"""
Script Camera Client - Capture ảnh từ camera và gửi lên Server AI Flask
Chạy script này trên PC có camera để gửi ảnh lên server nhận diện
"""
import cv2
import base64
import requests
import time
import argparse
from datetime import datetime

# Cấu hình mặc định
DEFAULT_AI_SERVER = "http://localhost:8000"
DEFAULT_CAMERA_ID = 0  # 0 = webcam mặc định, có thể đổi thành URL RTSP
DEFAULT_INTERVAL = 2  # Gửi ảnh mỗi 2 giây
DEFAULT_FACE_DETECT_INTERVAL = 0.5  # Kiểm tra khuôn mặt mỗi 0.5 giây

# Load cascade để phát hiện khuôn mặt (tùy chọn, để chỉ gửi khi có mặt)
face_cascade = None
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except:
    print("⚠️ Không thể load face cascade, sẽ gửi tất cả ảnh")


def capture_and_send(camera_id, ai_server_url, interval, only_when_face_detected=True):
    """
    Capture ảnh từ camera và gửi lên server AI
    
    Args:
        camera_id: ID camera (0, 1, 2...) hoặc URL RTSP
        ai_server_url: URL của Flask server (vd: http://192.168.1.100:8000)
        interval: Khoảng thời gian giữa các lần gửi (giây)
        only_when_face_detected: Chỉ gửi khi phát hiện khuôn mặt
    """
    print(f"[CAMERA] Dang mo camera {camera_id}...")
    
    # Mở camera
    if isinstance(camera_id, str) and camera_id.startswith('rtsp://'):
        cap = cv2.VideoCapture(camera_id)
    else:
        cap = cv2.VideoCapture(int(camera_id))
    
    if not cap.isOpened():
        print(f"[ERROR] Khong the mo camera {camera_id}")
        return
    
    print(f"[OK] Camera da san sang")
    print(f"[INFO] Dang gui anh toi: {ai_server_url}/recog")
    print(f"[INFO] Interval: {interval} giay")
    print("Nhan phim 'q' de thoat\n")
    
    last_send_time = 0
    frame_count = 0
    last_recognized_name = "Unknown"
    last_recognized_time = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Khong doc duoc frame tu camera")
                time.sleep(1)
                continue
            
            frame_count += 1
            current_time = time.time()
            
            # Kiểm tra xem có khuôn mặt không (nếu bật)
            has_face = True
            faces = []
            if face_cascade is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                if only_when_face_detected:
                    has_face = len(faces) > 0
            
            # Gửi ảnh nếu đã đủ thời gian và có khuôn mặt
            if has_face and (current_time - last_send_time >= interval):
                # Resize frame để giảm dung lượng
                height, width = frame.shape[:2]
                max_size = 800
                if width > max_size or height > max_size:
                    scale = max_size / max(width, height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame_resized = cv2.resize(frame, (new_width, new_height))
                else:
                    frame_resized = frame
                    new_width, new_height = width, height
                
                # Encode ảnh thành JPEG
                _, buffer = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Gửi POST request tới Flask server
                try:
                    response = requests.post(
                        f"{ai_server_url}/recog",
                        data={
                            'image': img_base64,
                            'w': new_width,
                            'h': new_height
                        },
                        timeout=5
                    )
                    
                    if response.status_code == 200:
                        recognized_name = response.text.strip() or "Unknown"
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        last_recognized_name = recognized_name
                        last_recognized_time = current_time
                        if recognized_name != "Unknown":
                            print(f"[{timestamp}] [OK] Nhan dien: {recognized_name}")
                        else:
                            print(f"[{timestamp}] [UNKNOWN] Khong nhan dien duoc")
                    else:
                        print(f"[WARNING] Server tra ve loi: {response.status_code}")
                        
                except requests.exceptions.RequestException as e:
                    print(f"[ERROR] Loi ket noi toi server: {e}")
                
                last_send_time = current_time
            
            # Vẽ vòng và tên người trên frame
            if len(faces) > 0:
                color = (0, 255, 0) if last_recognized_name != "Unknown" else (0, 0, 255)
                for (x, y, w, h) in faces:
                    center = (x + w // 2, y + h // 2)
                    radius = max(w, h) // 2
                    cv2.circle(frame, center, radius, color, 2)
                    label = last_recognized_name
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, color, 2, cv2.LINE_AA)
            else:
                if current_time - last_recognized_time > 3:
                    last_recognized_name = "Unknown"
            
            # Hiển thị preview (tùy chọn)
            cv2.imshow('Camera - Nhấn Q để thoát', frame)
            
            # Thoát khi nhấn 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n[STOP] Dung boi nguoi dung")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[OK] Da dong camera")


def main():
    parser = argparse.ArgumentParser(description='Camera Client - Gửi ảnh lên Server AI')
    parser.add_argument('--camera', type=str, default=str(DEFAULT_CAMERA_ID),
                        help='Camera ID (0, 1, 2...) hoặc URL RTSP (vd: rtsp://192.168.1.100:554/stream)')
    parser.add_argument('--server', type=str, default=DEFAULT_AI_SERVER,
                        help=f'URL của Flask AI server (mặc định: {DEFAULT_AI_SERVER})')
    parser.add_argument('--interval', type=float, default=DEFAULT_INTERVAL,
                        help=f'Khoảng thời gian giữa các lần gửi (giây, mặc định: {DEFAULT_INTERVAL})')
    parser.add_argument('--always-send', action='store_true',
                        help='Gửi tất cả ảnh, không chỉ khi có khuôn mặt')
    
    args = parser.parse_args()
    
    # Xử lý camera_id
    try:
        camera_id = int(args.camera)
    except ValueError:
        camera_id = args.camera  # URL RTSP
    
    capture_and_send(
        camera_id=camera_id,
        ai_server_url=args.server,
        interval=args.interval,
        only_when_face_detected=not args.always_send
    )


if __name__ == '__main__':
    main()


