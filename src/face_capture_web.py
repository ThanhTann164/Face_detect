"""
Face Capture Web API - API thu thập ảnh trên web với quality check
Tối ưu để lấy ảnh chất lượng cao, không trùng lặp
"""
import os
import sys
import numpy as np
import cv2
import base64
import time
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import hashlib

# Import pipeline
sys.path.insert(0, os.path.dirname(__file__))
from face_pipeline_unified import UnifiedFacePipeline

class FaceCaptureWeb:
    """
    Quản lý thu thập ảnh trên web với quality check
    """
    
    def __init__(self, facenet_model_path: str, gpu_memory_fraction: float = 0.6):
        """
        Khởi tạo capture manager
        
        Args:
            facenet_model_path: Đường dẫn FaceNet model
            gpu_memory_fraction: Tỷ lệ GPU memory
        """
        self.pipeline = UnifiedFacePipeline(facenet_model_path, gpu_memory_fraction)
        
        # Cache để tránh trùng lặp
        self.embedding_cache = {}  # {hash: embedding} để so sánh similarity
        self.min_time_interval = 0.5  # Tối thiểu 0.5 giây giữa các ảnh
        self.last_capture_time = {}
        
        # Quality thresholds
        self.min_face_size = 50
        self.min_blur_variance = 100
        self.min_detection_confidence = 0.7
        self.min_similarity_diff = 0.1  # Tối thiểu khác biệt với ảnh trước
        
        print("[OK] Face Capture Web initialized")
    
    def capture_from_base64(self, 
                           base64_image: str,
                           person_name: str = None,
                           min_interval: float = 0.5) -> Dict:
        """
        Capture ảnh từ base64 string
        
        Args:
            base64_image: Base64 encoded image string
            person_name: Tên người (để cache theo người)
            min_interval: Thời gian tối thiểu giữa các capture (giây)
        
        Returns:
            result: Dict với kết quả
        """
        result = {
            'success': False,
            'message': '',
            'image_saved': False,
            'quality_score': 0.0,
            'stats': {}
        }
        
        try:
            # 1. Decode base64
            try:
                image_data = base64.b64decode(base64_image.split(',')[-1])
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is None:
                    result['message'] = 'Failed to decode image'
                    return result
            except Exception as e:
                result['message'] = f'Failed to decode base64: {str(e)}'
                return result
            
            # 2. Kiểm tra time interval
            current_time = time.time()
            if person_name:
                last_time = self.last_capture_time.get(person_name, 0)
                if current_time - last_time < min_interval:
                    result['message'] = f'Too soon: {current_time - last_time:.2f}s < {min_interval}s'
                    return result
            
            # 3. Kiểm tra chất lượng
            is_valid, reason = self.pipeline.check_image_quality(
                image,
                min_face_size=self.min_face_size
            )
            
            if not is_valid:
                result['message'] = f'Quality check failed: {reason}'
                result['stats']['quality_reason'] = reason
                return result
            
            # 4. Xử lý để lấy embedding
            embedding, info = self.pipeline.process_image(image, return_face_image=True)
            
            if embedding is None:
                error = info.get('error', 'Unknown error')
                result['message'] = f'Processing failed: {error}'
                return result
            
            # 5. Kiểm tra trùng lặp với ảnh trước
            if person_name and person_name in self.embedding_cache:
                prev_embedding = self.embedding_cache[person_name]
                
                # Tính cosine similarity
                similarity = np.dot(embedding, prev_embedding) / (
                    np.linalg.norm(embedding) * np.linalg.norm(prev_embedding)
                )
                
                if similarity > (1.0 - self.min_similarity_diff):
                    result['message'] = f'Image too similar to previous: {similarity:.4f}'
                    result['stats']['similarity'] = float(similarity)
                    return result
            
            # 6. Tính quality score
            quality_score = self._calculate_quality_score(image, info, embedding)
            
            # 7. Cập nhật cache và time
            if person_name:
                self.embedding_cache[person_name] = embedding
                self.last_capture_time[person_name] = current_time
            
            # 8. Kết quả thành công
            result['success'] = True
            result['message'] = 'Image captured successfully'
            result['image_saved'] = True
            result['quality_score'] = quality_score
            result['stats'] = {
                'detection_confidence': info.get('confidence', 0),
                'face_size': self._get_face_size(info.get('bounding_box')),
                'timestamp': datetime.now().isoformat()
            }
            
            return result
        
        except Exception as e:
            result['message'] = f'Exception: {str(e)}'
            return result
    
    def _calculate_quality_score(self, image: np.ndarray, info: Dict, embedding: np.ndarray) -> float:
        """
        Tính quality score cho ảnh (0.0 - 1.0)
        
        Args:
            image: numpy array
            info: Dict với thông tin detection
            embedding: numpy array embedding
        
        Returns:
            score: Quality score
        """
        score = 0.0
        
        # 1. Detection confidence (0-0.4)
        detection_conf = info.get('confidence', 0)
        score += detection_conf * 0.4
        
        # 2. Face size (0-0.3)
        face_size = self._get_face_size(info.get('bounding_box'))
        if face_size > 100:
            score += 0.3
        elif face_size > 50:
            score += 0.15
        
        # 3. Blur check (0-0.3)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var > 200:
            score += 0.3
        elif laplacian_var > 100:
            score += 0.15
        
        return min(1.0, score)
    
    def _get_face_size(self, bounding_box: Optional[List]) -> int:
        """Lấy kích thước khuôn mặt từ bounding box"""
        if bounding_box is None or len(bounding_box) < 4:
            return 0
        width = bounding_box[2] - bounding_box[0]
        height = bounding_box[3] - bounding_box[1]
        return int(min(width, height))
    
    def save_image_to_raw(self, image: np.ndarray, person_name: str, raw_data_dir: str) -> Tuple[bool, str]:
        """
        Lưu ảnh vào thư mục raw sau khi đã quality check
        
        Args:
            image: numpy array (BGR format từ OpenCV)
            person_name: Tên người
            raw_data_dir: Thư mục raw data gốc
        
        Returns:
            success: Thành công hay không
            filepath: Đường dẫn file đã lưu
        """
        person_name = person_name.lower().strip()
        person_dir = os.path.join(raw_data_dir, person_name)
        os.makedirs(person_dir, exist_ok=True)
        
        filename = f"{person_name}_{int(time.time()*1000)}.jpg"
        filepath = os.path.join(person_dir, filename)
        
        try:
            cv2.imwrite(filepath, image)
            return True, filepath
        except Exception as e:
            return False, str(e)
    
    def capture_and_save(self, 
                        base64_image: str,
                        person_name: str,
                        raw_data_dir: str,
                        min_interval: float = 0.5) -> Dict:
        """
        Capture và lưu ảnh vào thư mục raw (all-in-one)
        
        Args:
            base64_image: Base64 encoded image
            person_name: Tên người
            raw_data_dir: Thư mục raw data
            min_interval: Thời gian tối thiểu giữa các capture
        
        Returns:
            result: Dict với kết quả
        """
        result = self.capture_from_base64(base64_image, person_name, min_interval)
        
        if result['success'] and result['image_saved']:
            # Decode lại ảnh để lưu
            try:
                image_data = base64.b64decode(base64_image.split(',')[-1])
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is not None:
                    success, filepath = self.save_image_to_raw(image, person_name, raw_data_dir)
                    if success:
                        result['filepath'] = filepath
                        result['message'] = f"Đã lưu ảnh vào {filepath}"
                    else:
                        result['success'] = False
                        result['message'] = f"Không thể lưu ảnh: {filepath}"
            except Exception as e:
                result['success'] = False
                result['message'] = f"Lỗi khi decode/lưu ảnh: {str(e)}"
        
        return result
    
    def reset_cache(self, person_name: str = None):
        """Reset cache cho một người hoặc tất cả"""
        if person_name:
            if person_name in self.embedding_cache:
                del self.embedding_cache[person_name]
            if person_name in self.last_capture_time:
                del self.last_capture_time[person_name]
        else:
            self.embedding_cache.clear()
            self.last_capture_time.clear()
    
    def close(self):
        """Đóng resources"""
        self.pipeline.close()

