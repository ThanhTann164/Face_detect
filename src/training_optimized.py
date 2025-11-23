"""
Training Module - Train classifier với pipeline nhất quán
Đảm bảo preprocessing giống hệt inference
"""
import os
import sys
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
import cv2
import imageio

# Import các module mới
sys.path.insert(0, os.path.dirname(__file__))
from face_detect import FaceDetector
from face_align import FaceAligner
from face_embedding import FaceEmbedder
from face_compare import FaceComparator

class FaceTrainer:
    """
    Trainer với pipeline nhất quán
    """
    
    def __init__(self, 
                 facenet_model_path,
                 data_dir,
                 output_model_path,
                 image_size=160,
                 margin=0,
                 use_landmarks=True):
        """
        Khởi tạo trainer
        
        Args:
            facenet_model_path: Đường dẫn FaceNet model
            data_dir: Thư mục chứa ảnh đã align (processed)
            output_model_path: Đường dẫn lưu model
            image_size: Kích thước ảnh (160)
            margin: Margin khi align (0 = không margin)
            use_landmarks: Có dùng landmarks alignment hay không
        """
        self.facenet_model_path = facenet_model_path
        self.data_dir = data_dir
        self.output_model_path = output_model_path
        self.image_size = image_size
        self.margin = margin
        self.use_landmarks = use_landmarks
        
        # Khởi tạo các module
        print("[INFO] Initializing face detector...")
        self.detector = FaceDetector()
        
        print("[INFO] Initializing face aligner...")
        self.aligner = FaceAligner(image_size=image_size, margin=margin)
        
        print("[INFO] Initializing face embedder...")
        self.embedder = FaceEmbedder(facenet_model_path, image_size=image_size)
        
        print("[OK] Trainer initialized")
    
    def load_dataset(self):
        """
        Load dataset từ thư mục processed
        Mỗi thư mục con là một class (người)
        
        Returns:
            images: list of numpy arrays
            labels: list of labels (tên người)
            class_names: list of unique class names
        """
        images = []
        labels = []
        class_names = []
        
        if not os.path.isdir(self.data_dir):
            raise ValueError(f"Data directory not found: {self.data_dir}")
        
        # Lấy danh sách các class (thư mục con)
        for class_name in sorted(os.listdir(self.data_dir)):
            class_dir = os.path.join(self.data_dir, class_name)
            
            if not os.path.isdir(class_dir):
                continue
            
            # Lấy tất cả ảnh trong thư mục
            image_files = [
                f for f in os.listdir(class_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            
            if len(image_files) == 0:
                print(f"[WARNING] No images found in {class_dir}")
                continue
            
            print(f"[INFO] Loading {len(image_files)} images for class: {class_name}")
            
            # Load từng ảnh
            for img_file in image_files:
                img_path = os.path.join(class_dir, img_file)
                try:
                    # Đọc ảnh
                    img = imageio.imread(img_path)
                    
                    # Chuyển sang RGB nếu cần
                    if img.ndim == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    elif img.shape[2] == 4:
                        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                    
                    # Đảm bảo là uint8
                    if img.dtype != np.uint8:
                        if img.max() <= 1.0:
                            img = (img * 255).astype(np.uint8)
                        else:
                            img = img.astype(np.uint8)
                    
                    images.append(img)
                    labels.append(len(class_names))  # Label là index
                except Exception as e:
                    print(f"[ERROR] Failed to load {img_path}: {e}")
                    continue
            
            class_names.append(class_name)
        
        print(f"[OK] Loaded {len(images)} images from {len(class_names)} classes")
        return images, labels, class_names
    
    def extract_embeddings(self, images, batch_size=32):
        """
        Extract embeddings từ ảnh
        Ảnh trong processed đã được align và resize, chỉ cần prewhiten
        
        Args:
            images: list of numpy arrays (đã align, 160x160)
            batch_size: Batch size khi extract
        
        Returns:
            embeddings: numpy array (N, embedding_size)
        """
        embeddings = []
        
        print(f"[INFO] Extracting embeddings from {len(images)} images...")
        print(f"[INFO] Note: Images are already aligned and resized to {self.image_size}x{self.image_size}")
        
        # Process từng batch
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_embeddings = []
            
            for img in batch_images:
                try:
                    # Ảnh đã được align và resize, chỉ cần prewhiten và extract
                    # Đảm bảo là RGB và uint8
                    if img.dtype != np.uint8:
                        if img.max() <= 1.0:
                            img = (img * 255).astype(np.uint8)
                        else:
                            img = img.astype(np.uint8)
                    
                    # Resize về image_size nếu cần (thường không cần vì đã resize)
                    if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
                        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
                    
                    # Extract embedding (sẽ prewhiten bên trong)
                    embedding = self.embedder.get_embedding(img, normalize=True)
                    batch_embeddings.append(embedding)
                except Exception as e:
                    print(f"[ERROR] Failed to extract embedding: {e}")
                    import traceback
                    traceback.print_exc()
                    # Tạo embedding zero - lấy size từ embedding đầu tiên
                    if len(batch_embeddings) > 0:
                        embedding_size = len(batch_embeddings[0])
                    else:
                        # Fallback: dùng size mặc định của FaceNet
                        embedding_size = 512
                    batch_embeddings.append(np.zeros(embedding_size))
            
            embeddings.extend(batch_embeddings)
            
            if (i + batch_size) % 100 == 0 or i + batch_size >= len(images):
                print(f"[INFO] Processed {min(i+batch_size, len(images))}/{len(images)} images")
        
        embeddings = np.array(embeddings)
        print(f"[OK] Extracted {embeddings.shape[0]} embeddings (size: {embeddings.shape[1]})")
        
        return embeddings
    
    def train_classifier(self, embeddings, labels, use_svm=True, use_normalization=False):
        """
        Train classifier
        
        Args:
            embeddings: numpy array (N, embedding_size)
            labels: list hoặc array (N,)
            use_svm: Dùng SVM hay không (nếu False, dùng cosine similarity)
            use_normalization: Có normalize embeddings trước khi train SVM không
        
        Returns:
            model: Trained model
            scaler: StandardScaler (nếu dùng normalization)
        """
        labels = np.array(labels)
        
        if use_svm:
            print("[INFO] Training SVM classifier...")
            
            # Normalize embeddings nếu cần
            scaler = None
            if use_normalization:
                scaler = StandardScaler()
                embeddings = scaler.fit_transform(embeddings)
                print("[OK] Normalized embeddings")
            
            # Train SVM
            model = SVC(kernel='linear', probability=True, C=1.0)
            model.fit(embeddings, labels)
            
            print("[OK] SVM classifier trained")
            return model, scaler
        else:
            # Không dùng SVM, chỉ lưu embeddings và labels
            print("[INFO] Using cosine similarity (no classifier)")
            return {'embeddings': embeddings, 'labels': labels}, None
    
    def save_model(self, model, class_names, scaler=None):
        """
        Lưu model
        
        Args:
            model: Trained model
            class_names: List of class names
            scaler: StandardScaler (nếu có)
        """
        save_data = {
            'model': model,
            'class_names': class_names,
            'scaler': scaler,
            'image_size': self.image_size,
            'margin': self.margin,
            'use_landmarks': self.use_landmarks
        }
        
        with open(self.output_model_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"[OK] Model saved to {self.output_model_path}")
    
    def train(self, use_svm=True, use_normalization=False, test_split=0.2):
        """
        Train toàn bộ pipeline
        
        Args:
            use_svm: Dùng SVM hay cosine similarity
            use_normalization: Có normalize embeddings không
            test_split: Tỷ lệ test set
        """
        # Load dataset
        images, labels, class_names = self.load_dataset()
        
        if len(images) == 0:
            raise ValueError("No images found in dataset!")
        
        # Extract embeddings
        embeddings = self.extract_embeddings(images)
        
        # Split train/test
        if test_split > 0:
            X_train, X_test, y_train, y_test = train_test_split(
                embeddings, labels, test_size=test_split, random_state=42, stratify=labels
            )
            print(f"[INFO] Train: {len(X_train)}, Test: {len(X_test)}")
        else:
            X_train, y_train = embeddings, labels
            X_test, y_test = None, None
        
        # Train classifier
        model, scaler = self.train_classifier(X_train, y_train, use_svm, use_normalization)
        
        # Evaluate
        if X_test is not None and use_svm:
            predictions = model.predict(X_test)
            accuracy = np.mean(predictions == y_test)
            print(f"[INFO] Test accuracy: {accuracy:.4f}")
            
            # Classification report
            print("\nClassification Report:")
            print(classification_report(y_test, predictions, target_names=class_names))
        
        # Save model
        self.save_model(model, class_names, scaler)
        
        # Cleanup
        self.detector.close()
        self.embedder.close()
        
        return model, class_names, scaler
    
    def close(self):
        """Đóng các resources"""
        self.detector.close()
        self.embedder.close()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train face recognition model')
    parser.add_argument('--facenet_model', type=str, required=True,
                       help='Path to FaceNet .pb model')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to processed dataset directory')
    parser.add_argument('--output', type=str, required=True,
                       help='Output model path (.pkl)')
    parser.add_argument('--image_size', type=int, default=160,
                       help='Image size (default: 160)')
    parser.add_argument('--margin', type=int, default=0,
                       help='Margin pixels (default: 0)')
    parser.add_argument('--use_svm', action='store_true',
                       help='Use SVM classifier')
    parser.add_argument('--normalize', action='store_true',
                       help='Normalize embeddings before SVM')
    parser.add_argument('--test_split', type=float, default=0.2,
                       help='Test split ratio (default: 0.2)')
    
    args = parser.parse_args()
    
    trainer = FaceTrainer(
        facenet_model_path=args.facenet_model,
        data_dir=args.data_dir,
        output_model_path=args.output,
        image_size=args.image_size,
        margin=args.margin
    )
    
    try:
        trainer.train(
            use_svm=args.use_svm,
            use_normalization=args.normalize,
            test_split=args.test_split
        )
        print("\n[OK] Training completed successfully!")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        trainer.close()

