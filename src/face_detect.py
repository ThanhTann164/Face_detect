"""
Face Detection Module - Sử dụng MTCNN
Đảm bảo detection nhất quán giữa training và inference
"""
import cv2
import numpy as np
import tensorflow as tf
import sys
import os

# Thêm đường dẫn để import align module
sys.path.insert(0, os.path.dirname(__file__))
import align.detect_face

class FaceDetector:
    """Face detector sử dụng MTCNN - nhất quán cho training và inference"""
    
    def __init__(self, gpu_memory_fraction=0.6):
        """
        Khởi tạo MTCNN detector
        
        Args:
            gpu_memory_fraction: Tỷ lệ GPU memory sử dụng
        """
        self.minsize = 20
        self.threshold = [0.6, 0.7, 0.7]  # P-Net, R-Net, O-Net thresholds
        self.factor = 0.709  # Scale factor
        self.gpu_memory_fraction = gpu_memory_fraction
        
        # Tạo graph và session cho MTCNN
        self.graph = tf.Graph()
        self.sess = None
        self.pnet = None
        self.rnet = None
        self.onet = None
        
        self._init_mtcnn()
    
    def _init_mtcnn(self):
        """Khởi tạo MTCNN networks"""
        with self.graph.as_default():
            gpu_options = tf.compat.v1.GPUOptions(
                per_process_gpu_memory_fraction=self.gpu_memory_fraction
            )
            config = tf.compat.v1.ConfigProto(
                gpu_options=gpu_options,
                log_device_placement=False
            )
            self.sess = tf.compat.v1.Session(config=config)
            
            with self.sess.as_default():
                # Tạo MTCNN networks
                self.pnet, self.rnet, self.onet = align.detect_face.create_mtcnn(
                    self.sess, 
                    os.path.join(os.path.dirname(__file__), "align")
                )
        
        print("[OK] Face detector (MTCNN) initialized")
    
    def detect(self, image, min_face_size=None):
        """
        Detect faces trong ảnh
        
        Args:
            image: numpy array (H, W, 3) - RGB image
            min_face_size: Minimum face size (default: self.minsize)
        
        Returns:
            bounding_boxes: numpy array (N, 5) - [x1, y1, x2, y2, confidence]
            landmarks: numpy array (N, 10) - [x1, y1, x2, y2, ..., x5, y5] (5 landmarks)
        """
        if min_face_size is None:
            min_face_size = self.minsize
        
        # Chuyển BGR sang RGB nếu cần
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Kiểm tra xem có phải BGR không (OpenCV format)
            if image.dtype == np.uint8:
                # Giả sử là BGR, chuyển sang RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
        else:
            image_rgb = image
        
        # Detect faces
        with self.graph.as_default():
            with self.sess.as_default():
                bounding_boxes, landmarks = align.detect_face.detect_face(
                    image_rgb,
                    min_face_size,
                    self.pnet,
                    self.rnet,
                    self.onet,
                    self.threshold,
                    self.factor
                )
        
        return bounding_boxes, landmarks
    
    def get_largest_face(self, bounding_boxes, landmarks):
        """
        Chọn khuôn mặt lớn nhất từ kết quả detection
        
        Args:
            bounding_boxes: numpy array (N, 5)
            landmarks: numpy array (N, 10)
        
        Returns:
            best_box: numpy array (5,) - [x1, y1, x2, y2, confidence]
            best_landmarks: numpy array (10,) - 5 landmarks
        """
        if bounding_boxes.shape[0] == 0:
            return None, None
        
        # Tính diện tích các khuôn mặt
        areas = (bounding_boxes[:, 2] - bounding_boxes[:, 0]) * \
                (bounding_boxes[:, 3] - bounding_boxes[:, 1])
        
        # Chọn khuôn mặt lớn nhất
        largest_idx = np.argmax(areas)
        
        best_box = bounding_boxes[largest_idx]
        best_landmarks = landmarks[largest_idx] if landmarks is not None else None
        
        return best_box, best_landmarks
    
    def close(self):
        """Đóng session"""
        if self.sess is not None:
            self.sess.close()


