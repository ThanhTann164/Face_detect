"""
Script chụp ảnh từ camera và lưu vào dataset
Dùng để thu thập ảnh khuôn mặt cho người mới
"""
import cv2
import os
import sys
import time
from datetime import datetime

# Đường dẫn dataset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "Dataset", "FaceData", "raw")

def capture_images(person_name, num_images=100, camera_id=0, delay=0.5, auto_mode=True):
    """
    Chụp ảnh từ camera và lưu vào dataset (tự động hoặc thủ công)
    
    Args:
        person_name: Tên người (sẽ tạo thư mục với tên này)
        num_images: Số lượng ảnh cần chụp
        camera_id: ID camera (0, 1, 2...)
        delay: Thời gian delay giữa các lần chụp (giây) - chỉ dùng khi auto_mode=True
        auto_mode: True = tự động chụp, False = chụp thủ công bằng phím
    """
    import time
    
    # Tạo thư mục
    person_dir = os.path.join(RAW_DIR, person_name.lower().replace(' ', '_'))
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)
        print(f"[OK] Da tao thu muc: {person_dir}")
    else:
        print(f"[INFO] Thu muc da ton tai: {person_dir}")
    
    # Đếm số ảnh hiện có
    existing_images = [f for f in os.listdir(person_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    start_count = len(existing_images)
    
    print("\n" + "=" * 60)
    print("CHUP ANH KHUON MAT TU CAMERA")
    print("=" * 60)
    print(f"\n[INFO] Ten nguoi: {person_name}")
    print(f"[INFO] So anh hien co: {start_count}")
    print(f"[INFO] So anh se chup: {num_images}")
    print(f"[INFO] Thu muc luu: {person_dir}")
    if auto_mode:
        print(f"[INFO] Che do: TU DONG (delay: {delay} giay giua moi lan chup)")
        print(f"[INFO] Nhan 'q' de dung som")
    else:
        print("\n[HUONG DAN]")
        print("  - Nhan SPACE hoac 's' de chup anh")
        print("  - Nhan 'q' de thoat")
        print("  - Nhan 'r' de reset dem")
    print("=" * 60 + "\n")
    
    # Mở camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[ERROR] Khong the mo camera {camera_id}")
        print(f"[INFO] Thu doi camera_id khac: 1, 2, 3...")
        return False
    
    print(f"[OK] Camera da san sang")
    
    # Đợi 2 giây để người dùng chuẩn bị
    if auto_mode:
        print(f"[INFO] Chuan bi trong 3 giay...")
        for i in range(3, 0, -1):
            print(f"[INFO] {i}...")
            time.sleep(1)
        print(f"[INFO] Bat dau chup anh tu dong!\n")
    
    count = start_count
    frame_count = 0
    last_capture_time = 0
    
    try:
        while count < start_count + num_images:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Khong doc duoc frame")
                continue
            
            frame_count += 1
            
            # Flip frame để dễ nhìn (như gương)
            frame = cv2.flip(frame, 1)
            
            current_time = time.time()
            
            # Chế độ tự động: chụp theo delay
            if auto_mode:
                if current_time - last_capture_time >= delay:
                    # Chụp ảnh
                    timestamp = int(datetime.now().timestamp() * 1000)
                    filename = f"{timestamp}.jpg"
                    filepath = os.path.join(person_dir, filename)
                    
                    # Lưu ảnh
                    cv2.imwrite(filepath, frame)
                    count += 1
                    last_capture_time = current_time
                    print(f"[OK] Da chup anh {count}/{start_count + num_images}: {filename}")
                    
                    # Hiển thị thông báo trên frame
                    h, w = frame.shape[:2]
                    cv2.putText(frame, f"DA CHUP! ({count}/{start_count + num_images})", 
                               (w//2 - 200, h//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                else:
                    # Hiển thị đếm ngược
                    h, w = frame.shape[:2]
                    remaining = delay - (current_time - last_capture_time)
                    countdown = f"Chup sau: {remaining:.1f}s - Anh {count + 1}/{start_count + num_images}"
                    cv2.putText(frame, countdown, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                # Chế độ thủ công
                info_text = f"Anh {count + 1}/{start_count + num_images} - Nhan SPACE de chup, 'q' de thoat"
                cv2.putText(frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Hiển thị khung để người dùng căn chỉnh
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (w//4, h//4), (3*w//4, 3*h//4), (0, 255, 0), 2)
            cv2.putText(frame, "Dat khuon mat trong khung", (w//4, h//4 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Hiển thị frame
            window_name = 'Chup Anh Tu Dong - Nhan q de dung' if auto_mode else 'Chup Anh - Nhan SPACE de chup, q de thoat'
            cv2.imshow(window_name, frame)
            
            # Xử lý phím (luôn cho phép nhấn 'q' để dừng)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n[INFO] Dung chup anh")
                break
            elif not auto_mode:
                # Chế độ thủ công
                if key == ord(' ') or key == ord('s'):
                    # Chụp ảnh
                    timestamp = int(datetime.now().timestamp() * 1000)
                    filename = f"{timestamp}.jpg"
                    filepath = os.path.join(person_dir, filename)
                    
                    # Lưu ảnh
                    cv2.imwrite(filepath, frame)
                    count += 1
                    print(f"[OK] Da chup anh {count}: {filename}")
                    
                    # Hiển thị thông báo
                    cv2.putText(frame, "DA CHUP!", (w//2 - 100, h//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    cv2.imshow(window_name, frame)
                    cv2.waitKey(500)  # Hiển thị 0.5 giây
                elif key == ord('r'):
                    # Reset đếm
                    count = start_count
                    print(f"[INFO] Reset dem ve {count}")
    
    except KeyboardInterrupt:
        print("\n[INFO] Dung boi nguoi dung")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Thống kê
        final_images = [f for f in os.listdir(person_dir) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print("\n" + "=" * 60)
        print("HOAN TAT!")
        print("=" * 60)
        print(f"[INFO] Tong so anh da chup: {len(final_images)}")
        print(f"[INFO] Thu muc: {person_dir}")
        print(f"\n[BUOC TIEP THEO]")
        print(f"Chay script de align va train:")
        print(f"  cd src")
        print(f"  py add_new_person.py --name \"{person_name}\"")
        return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Chup anh khuon mat tu camera')
    parser.add_argument('--name', type=str, required=True,
                        help='Ten nguoi (dung chu thuong, khong dau)')
    parser.add_argument('--num', type=int, default=100,
                        help='So luong anh can chup (mac dinh: 100)')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera ID (mac dinh: 0)')
    parser.add_argument('--delay', type=float, default=0.5,
                        help='Thoi gian delay giua moi lan chup (giay, mac dinh: 0.5)')
    parser.add_argument('--manual', action='store_true',
                        help='Che do thu cong (nhan phim de chup, mac dinh: tu dong)')
    
    args = parser.parse_args()
    
    person_name = args.name.lower().replace(' ', '_')
    
    capture_images(person_name, args.num, args.camera, args.delay, auto_mode=not args.manual)

if __name__ == '__main__':
    main()

