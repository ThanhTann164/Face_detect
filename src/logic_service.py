"""
Logic Service - Xử lý điều kiện mở cửa
Service này subscribe kết quả từ AI và áp dụng logic trước khi gửi lệnh mở cửa
Có thể chạy riêng hoặc tích hợp vào Flask
"""
import paho.mqtt.client as mqtt
import ssl
import json
from datetime import datetime, time as dt_time
import logging

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cấu hình MQTT
MQTT_SERVER = "5867fe71cdee4ac0910debc62feddee7.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_USER = "nguyenluc0112"
MQTT_PASSWORD = "buithanhTan@123"

# Topics
TOPIC_AI_RESULT = "ai/result"  # Topic nhận kết quả từ AI
TOPIC_DOOR_CMD = "door/cmd"    # Topic gửi lệnh tới ESP32
TOPIC_DOOR_STATUS = "door/status"  # Topic nhận trạng thái từ ESP32
TOPIC_SENSORS = "fulloption"   # Topic nhận dữ liệu cảm biến

# Cấu hình logic
ALLOWED_NAMES = ["tan", "tan2"]  # Danh sách người được phép mở cửa
MIN_PROBABILITY = 0.8  # Xác suất tối thiểu để mở cửa

# Giờ mở cửa (có thể tùy chỉnh)
ALLOWED_HOURS_START = 6  # 6h sáng
ALLOWED_HOURS_END = 22   # 10h tối

# Biến trạng thái
door_status = "closed"
last_door_command = None


def is_time_allowed():
    """Kiểm tra xem có trong giờ cho phép mở cửa không"""
    now = datetime.now()
    current_hour = now.hour
    return ALLOWED_HOURS_START <= current_hour < ALLOWED_HOURS_END


def should_open_door(name, probability):
    """
    Logic quyết định có mở cửa không
    
    Args:
        name: Tên người được nhận diện
        probability: Xác suất nhận diện
        
    Returns:
        bool: True nếu nên mở cửa
    """
    # Kiểm tra tên có trong danh sách cho phép
    if name.lower() not in [n.lower() for n in ALLOWED_NAMES]:
        logger.warning(f"⚠️ Người không được phép: {name}")
        return False
    
    # Kiểm tra xác suất
    if probability < MIN_PROBABILITY:
        logger.warning(f"⚠️ Xác suất quá thấp: {probability:.2f}")
        return False
    
    # Kiểm tra giờ
    if not is_time_allowed():
        logger.warning(f"⚠️ Ngoài giờ cho phép mở cửa")
        return False
    
    # Kiểm tra cửa đã mở chưa
    if door_status == "open":
        logger.info("ℹ️ Cửa đã mở rồi")
        return False
    
    return True


def on_connect(client, userdata, flags, rc):
    """Callback khi kết nối MQTT"""
    if rc == 0:
        logger.info("✅ Đã kết nối MQTT broker")
        # Subscribe các topics
        client.subscribe(TOPIC_AI_RESULT, qos=1)
        client.subscribe(TOPIC_DOOR_STATUS, qos=1)
        client.subscribe(TOPIC_SENSORS, qos=0)  # Tùy chọn: đọc cảm biến
    else:
        logger.error(f"❌ Kết nối MQTT thất bại, mã lỗi: {rc}")


def on_message(client, userdata, msg):
    """Callback khi nhận message từ MQTT"""
    global door_status, last_door_command
    
    topic = msg.topic
    payload = msg.payload.decode('utf-8')
    
    try:
        if topic == TOPIC_AI_RESULT:
            # Nhận kết quả từ AI
            data = json.loads(payload)
            name = data.get('name', 'Unknown')
            probability = data.get('probability', 0.0)
            timestamp = data.get('timestamp', datetime.now().isoformat())
            
            logger.info(f"📥 Nhận kết quả AI: {name} (xác suất: {probability:.2f})")
            
            # Áp dụng logic
            if should_open_door(name, probability):
                logger.info(f"🚪 Gửi lệnh mở cửa cho {name}")
                client.publish(TOPIC_DOOR_CMD, "OPEN", qos=1)
                last_door_command = datetime.now()
            else:
                logger.info(f"🚫 Không mở cửa cho {name}")
        
        elif topic == TOPIC_DOOR_STATUS:
            # Nhận trạng thái cửa từ ESP32
            door_status = payload.lower()
            logger.info(f"🚪 Trạng thái cửa: {door_status}")
        
        elif topic == TOPIC_SENSORS:
            # Nhận dữ liệu cảm biến (có thể dùng để bổ sung logic)
            try:
                sensor_data = json.loads(payload)
                # Ví dụ: không mở cửa nếu có khí gas
                if sensor_data.get('gasAlert') == 'Danger':
                    logger.warning("⚠️ Phát hiện khí gas nguy hiểm, không mở cửa")
            except:
                pass
                
    except Exception as e:
        logger.error(f"❌ Lỗi xử lý message: {e}")


def main():
    """Hàm main để chạy logic service"""
    logger.info("🚀 Khởi động Logic Service...")
    logger.info(f"📋 Danh sách người được phép: {', '.join(ALLOWED_NAMES)}")
    logger.info(f"⏰ Giờ mở cửa: {ALLOWED_HOURS_START}h - {ALLOWED_HOURS_END}h")
    
    # Tạo MQTT client
    client = mqtt.Client(client_id="LogicService")
    client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
    client.tls_set(cert_reqs=ssl.CERT_REQUIRED)
    client.tls_insecure_set(True)  # Đặt False nếu có CA certificate
    
    # Set callbacks
    client.on_connect = on_connect
    client.on_message = on_message
    
    # Kết nối và chạy loop
    try:
        client.connect(MQTT_SERVER, MQTT_PORT, keepalive=60)
        logger.info("🔄 Đang chạy logic service...")
        client.loop_forever()
    except KeyboardInterrupt:
        logger.info("⏹️  Dừng service")
        client.disconnect()
    except Exception as e:
        logger.error(f"❌ Lỗi: {e}")


if __name__ == '__main__':
    main()


