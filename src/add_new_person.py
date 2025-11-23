"""
Script tự động để thêm người mới vào hệ thống nhận diện
Tự động align ảnh và train lại classifier
"""
import os
import sys
import subprocess
import shutil
from datetime import datetime

# Đường dẫn mặc định
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "Dataset", "FaceData", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "Dataset", "FaceData", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "Models")
FACENET_MODEL = os.path.join(MODEL_DIR, "20180402-114759", "20180402-114759.pb")
CLASSIFIER_MODEL = os.path.join(MODEL_DIR, "facemodel.pkl")

def print_step(step_num, message):
    """In thông báo bước"""
    print("\n" + "=" * 60)
    print(f"BUOC {step_num}: {message}")
    print("=" * 60)

def check_images_in_dir(dir_path):
    """Kiểm tra số lượng ảnh trong thư mục"""
    if not os.path.exists(dir_path):
        return 0
    images = [f for f in os.listdir(dir_path) 
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    return len(images)

def backup_model():
    """Backup model cũ"""
    if os.path.exists(CLASSIFIER_MODEL):
        backup_name = f"facemodel_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        backup_path = os.path.join(MODEL_DIR, backup_name)
        shutil.copy2(CLASSIFIER_MODEL, backup_path)
        print(f"[OK] Da backup model cu: {backup_path}")
        return True
    return False

def align_images(person_name):
    """Align ảnh khuôn mặt"""
    input_dir = os.path.join(RAW_DIR, person_name)
    output_dir = os.path.join(PROCESSED_DIR, person_name)
    
    if not os.path.exists(input_dir):
        print(f"[ERROR] Khong tim thay thu muc: {input_dir}")
        return False
    
    num_images = check_images_in_dir(input_dir)
    if num_images < 10:
        print(f"[WARNING] Chi co {num_images} anh. Khuyen nghi it nhat 10-20 anh")
        print("[INFO] Tiep tuc align tu dong (khong hoi y/n)")
    
    print(f"[INFO] Tim thay {num_images} anh trong {input_dir}")
    print(f"[INFO] Dang align anh...")
    
    # Chạy script align
    script_path = os.path.join(os.path.dirname(__file__), "align_dataset_mtcnn.py")
    cmd = [
        sys.executable,
        script_path,
        RAW_DIR,
        PROCESSED_DIR,
        "--image_size", "160",
        "--margin", "32",
        "--people", person_name
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("[OK] Align anh thanh cong!")
        
        # Kiểm tra kết quả
        num_processed = check_images_in_dir(output_dir)
        print(f"[INFO] Da xu ly {num_processed} anh")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Loi khi align: {e}")
        print(f"[ERROR] Output: {e.stdout}")
        print(f"[ERROR] Error: {e.stderr}")
        return False

def train_classifier():
    """Train lại classifier với tất cả người"""
    print(f"[INFO] Dang train classifier...")
    print(f"[INFO] Du lieu: {PROCESSED_DIR}")
    print(f"[INFO] Model: {FACENET_MODEL}")
    print(f"[INFO] Output: {CLASSIFIER_MODEL}")
    
    script_path = os.path.join(os.path.dirname(__file__), "classifier.py")
    cmd = [
        sys.executable,
        script_path,
        "TRAIN",
        PROCESSED_DIR,
        FACENET_MODEL,
        CLASSIFIER_MODEL,
        "--batch_size", "90",
        "--image_size", "160"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("[OK] Train classifier thanh cong!")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Loi khi train: {e}")
        print(f"[ERROR] Output: {e.stdout}")
        print(f"[ERROR] Error: {e.stderr}")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Them nguoi moi vao he thong nhan dien')
    parser.add_argument('--name', type=str, required=True,
                        help='Ten nguoi moi (dung chu thuong, khong dau, co the dung dau gach duoi)')
    parser.add_argument('--images_dir', type=str,
                        help='Duong dan thu muc chua anh (neu khac thu muc mac dinh)')
    parser.add_argument('--skip_align', action='store_true',
                        help='Bo qua buoc align (neu anh da duoc align)')
    parser.add_argument('--skip_train', action='store_true',
                        help='Bo qua buoc train (chi align)')
    
    args = parser.parse_args()
    
    person_name = args.name.lower().replace(' ', '_')
    
    print("\n" + "=" * 60)
    print("THEM NGUOI MOI VAO HE THONG NHAN DIEN")
    print("=" * 60)
    print(f"\n[INFO] Ten nguoi: {person_name}")
    
    # Bước 1: Kiểm tra và chuẩn bị
    print_step(1, "Kiem tra va chuan bi")
    
    if args.images_dir:
        # Copy ảnh từ thư mục chỉ định
        source_dir = args.images_dir
        target_dir = os.path.join(RAW_DIR, person_name)
        if os.path.exists(source_dir):
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            # Copy ảnh
            import glob
            images = glob.glob(os.path.join(source_dir, "*.jpg")) + \
                     glob.glob(os.path.join(source_dir, "*.jpeg")) + \
                     glob.glob(os.path.join(source_dir, "*.png"))
            for img in images:
                shutil.copy2(img, target_dir)
            print(f"[OK] Da copy {len(images)} anh vao {target_dir}")
        else:
            print(f"[ERROR] Khong tim thay thu muc: {source_dir}")
            return
    else:
        # Kiểm tra thư mục raw
        raw_person_dir = os.path.join(RAW_DIR, person_name)
        if not os.path.exists(raw_person_dir):
            print(f"[INFO] Tao thu muc: {raw_person_dir}")
            os.makedirs(raw_person_dir)
            print(f"[INFO] Vui long copy anh vao thu muc nay:")
            print(f"      {raw_person_dir}")
            print(f"[INFO] Sau do chay lai script nay")
            return
    
    # Bước 2: Align ảnh
    if not args.skip_align:
        print_step(2, "Align anh khuon mat")
        if not align_images(person_name):
            print("[ERROR] Align that bai!")
            return
    else:
        print("[INFO] Bo qua buoc align")
    
    # Bước 3: Backup model
    print_step(3, "Backup model cu")
    backup_model()
    
    # Bước 4: Train classifier
    if not args.skip_train:
        print_step(4, "Train lai classifier")
        if not train_classifier():
            print("[ERROR] Train that bai!")
            return
    else:
        print("[INFO] Bo qua buoc train")
    
    # Hoàn thành
    print("\n" + "=" * 60)
    print("HOAN TAT!")
    print("=" * 60)
    print(f"\n[OK] Da them nguoi '{person_name}' vao he thong")
    print(f"[INFO] Vui long restart AI server de ap dung model moi:")
    print(f"      cd src")
    print(f"      py face_rec_flask.py")
    print(f"\n[INFO] Test nhan dien:")
    print(f"      py test_face_recognition.py ../Dataset/FaceData/processed/{person_name}/face_001.png")

if __name__ == '__main__':
    main()

