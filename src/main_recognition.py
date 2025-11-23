"""
Main Recognition Module - Pipeline hoàn chỉnh cho nhận diện
Sử dụng pipeline nhất quán với training
"""
import os
import sys
import numpy as np
import cv2
import pickle
import json
from typing import Tuple, Optional, Dict

# Import các module
sys.path.insert(0, os.path.dirname(__file__))
from face_detect import FaceDetector
from face_align import FaceAligner
from face_embedding import FaceEmbedder
from face_compare import FaceComparator

class FaceRecognizer:
    """
    Face recognizer với pipeline nhất quán
    """
    
    def __init__(self,
                 facenet_model_path,
                 classifier_model_path,
                 image_size=160,
                 margin=0,
                 similarity_threshold=0.6,
                 method='cosine'):
        """
        Khởi tạo recognizer
        
        Args:
            facenet_model_path: Đường dẫn FaceNet model
            classifier_model_path: Đường dẫn classifier model (.pkl)
            image_size: Kích thước ảnh (160)
            margin: Margin khi align (0)
            similarity_threshold: Ngưỡng similarity để match
            method: 'cosine' hoặc 'euclidean'
        """
        self.facenet_model_path = facenet_model_path
        self.classifier_model_path = classifier_model_path
        self.image_size = image_size
        self.margin = margin
        
        # Khởi tạo các module
        print("[INFO] Initializing face detector...")
        self.detector = FaceDetector()
        
        print("[INFO] Initializing face aligner...")
        self.aligner = FaceAligner(image_size=image_size, margin=margin)
        
        print("[INFO] Initializing face embedder...")
        self.embedder = FaceEmbedder(facenet_model_path, image_size=image_size)
        
        print("[INFO] Loading classifier model...")
        self.model, self.class_names, self.scaler = self._load_classifier()
        
        print("[INFO] Initializing face comparator...")
        self.comparator = FaceComparator(method=method, threshold=similarity_threshold)
        
        print("[OK] Face recognizer initialized")
        print(f"[INFO] Loaded {len(self.class_names)} classes: {self.class_names}")
    
    def _load_classifier(self):
        """
        Load classifier model
        
        Returns:
            model: Trained model hoặc dict với embeddings
            class_names: List of class names
            scaler: StandardScaler hoặc None
        """
        with open(self.classifier_model_path, 'rb') as f:
            data = pickle.load(f)
        
        # Hỗ trợ cả format cũ và mới
        if isinstance(data, dict):
            model = data.get('model')
            class_names = data.get('class_names', [])
            scaler = data.get('scaler')
        elif isinstance(data, tuple):
            model, class_names = data
            scaler = None
        else:
            raise ValueError("Unknown model format")
        
        return model, class_names, scaler
    
    def recognize(self, image, return_details=False) -> Tuple[Optional[str], float, bool]:
        """
        Nhận diện khuôn mặt trong ảnh
        
        Args:
            image: numpy array (H, W, 3) - RGB hoặc BGR image
            return_details: Có trả về chi tiết hay không
        
        Returns:
            name: Tên người được nhận diện hoặc None
            confidence: Confidence score (0-1)
            is_match: Có match hay không
            details: Dict với chi tiết (nếu return_details=True)
        """
        # Chuyển BGR sang RGB nếu cần
        if len(image.shape) == 3:
            # Giả sử OpenCV format (BGR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # 1. Detect faces
        bounding_boxes, landmarks = self.detector.detect(image_rgb)
        
        if bounding_boxes.shape[0] == 0:
            if return_details:
                return None, 0.0, False, {'error': 'No face detected'}
            return None, 0.0, False
        
        # 2. Chọn khuôn mặt lớn nhất
        best_box, best_landmarks = self.detector.get_largest_face(bounding_boxes, landmarks)
        
        if best_box is None:
            if return_details:
                return None, 0.0, False, {'error': 'Failed to select face'}
            return None, 0.0, False
        
        # 3. Align face
        try:
            aligned_face = self.aligner.align_bbox(image_rgb, best_box, best_landmarks)
        except Exception as e:
            print(f"[ERROR] Alignment failed: {e}")
            if return_details:
                return None, 0.0, False, {'error': f'Alignment failed: {str(e)}'}
            return None, 0.0, False
        
        # 4. Extract embedding
        try:
            embedding = self.embedder.get_embedding(aligned_face, normalize=True)
        except Exception as e:
            print(f"[ERROR] Embedding extraction failed: {e}")
            if return_details:
                return None, 0.0, False, {'error': f'Embedding failed: {str(e)}'}
            return None, 0.0, False
        
        # 5. Normalize embedding nếu có scaler
        if self.scaler is not None:
            embedding = self.scaler.transform(embedding.reshape(1, -1))[0]
            # L2 normalize lại sau khi scale
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        
        # 6. Predict
        if isinstance(self.model, dict):
            # Dùng cosine similarity với database embeddings
            db_embeddings = self.model['embeddings']
            db_labels = self.model['labels']
            
            best_label, confidence, is_match = self.comparator.find_best_match(
                embedding, db_embeddings, db_labels
            )
            
            name = self.class_names[best_label] if best_label is not None else None
        else:
            # Dùng SVM classifier
            prediction = self.model.predict_proba(embedding.reshape(1, -1))[0]
            best_idx = np.argmax(prediction)
            confidence = float(prediction[best_idx])
            
            # Tối ưu: Threshold động cho SVM probability
            # SVM probability thường thấp hơn cosine similarity
            # Nên dùng threshold thấp hơn (0.3-0.4 thay vì 0.6-0.7)
            svm_threshold = max(0.25, self.comparator.threshold * 0.5)  # Giảm threshold cho SVM
            
            # Kiểm tra threshold
            is_match = confidence > svm_threshold
            name = self.class_names[best_idx] if is_match else None
            
            # Debug: print confidence
            print(f"[DEBUG] SVM prediction - Best: {self.class_names[best_idx]} ({confidence:.4f}), Threshold: {svm_threshold:.4f}, Match: {is_match}")
        
        if return_details:
            details = {
                'bounding_box': best_box.tolist(),
                'landmarks': best_landmarks.tolist() if best_landmarks is not None else None,
                'confidence': float(confidence),
                'is_match': is_match,
                'all_predictions': {}
            }
            
            # Thêm tất cả predictions nếu dùng SVM
            if not isinstance(self.model, dict):
                for i, prob in enumerate(prediction):
                    details['all_predictions'][self.class_names[i]] = float(prob)
            
            return name, confidence, is_match, details
        
        return name, confidence, is_match
    
    def recognize_batch(self, images):
        """
        Nhận diện batch ảnh
        
        Args:
            images: list of numpy arrays
        
        Returns:
            results: list of (name, confidence, is_match)
        """
        results = []
        for img in images:
            name, conf, match = self.recognize(img)
            results.append((name, conf, match))
        return results
    
    def close(self):
        """Đóng resources"""
        self.detector.close()
        self.embedder.close()

# Example usage
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Face recognition')
    parser.add_argument('--facenet_model', type=str, required=True,
                       help='Path to FaceNet .pb model')
    parser.add_argument('--classifier_model', type=str, required=True,
                       help='Path to classifier .pkl model')
    parser.add_argument('--image', type=str,
                       help='Path to test image')
    parser.add_argument('--threshold', type=float, default=0.6,
                       help='Similarity threshold (default: 0.6)')
    parser.add_argument('--method', type=str, default='cosine',
                       choices=['cosine', 'euclidean'],
                       help='Comparison method')
    
    args = parser.parse_args()
    
    # Khởi tạo recognizer
    recognizer = FaceRecognizer(
        facenet_model_path=args.facenet_model,
        classifier_model_path=args.classifier_model,
        similarity_threshold=args.threshold,
        method=args.method
    )
    
    try:
        if args.image:
            # Test với ảnh
            img = cv2.imread(args.image)
            if img is None:
                print(f"[ERROR] Failed to load image: {args.image}")
            else:
                name, confidence, is_match, details = recognizer.recognize(img, return_details=True)
                
                print(f"\n[RESULT]")
                print(f"  Name: {name}")
                print(f"  Confidence: {confidence:.4f}")
                print(f"  Match: {is_match}")
                print(f"\n[DETAILS]")
                print(json.dumps(details, indent=2))
        else:
            print("[INFO] Use --image to test recognition")
    except Exception as e:
        print(f"[ERROR] Recognition failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        recognizer.close()

