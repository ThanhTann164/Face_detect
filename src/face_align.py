"""
Face Alignment Module - 5-point landmark alignment
Chuẩn hóa khuôn mặt về cùng góc độ và kích thước
"""
import cv2
import numpy as np
try:
    from skimage import transform as trans
except ImportError:
    # Fallback nếu không có skimage
    print("[WARNING] skimage not found, using cv2 for alignment")
    trans = None

class FaceAligner:
    """
    Face aligner sử dụng 5-point landmarks
    Chuẩn hóa khuôn mặt về cùng góc độ và kích thước
    """
    
    # Reference landmarks cho 112x112 (ArcFace standard)
    # Hoặc 160x160 (FaceNet standard)
    REFERENCE_112 = np.array([
        [30.2946, 51.6963],  # Left eye
        [65.5318, 51.5014],  # Right eye
        [48.0252, 71.7366],  # Nose tip
        [33.5493, 92.3655],  # Left mouth corner
        [62.7299, 92.2041]   # Right mouth corner
    ], dtype=np.float32)
    
    # Reference landmarks cho 160x160
    REFERENCE_160 = REFERENCE_112 * (160.0 / 112.0)
    
    def __init__(self, image_size=160, margin=0):
        """
        Khởi tạo face aligner
        
        Args:
            image_size: Kích thước ảnh output (112 hoặc 160)
            margin: Margin pixels khi crop (0 = crop sát, >0 = thêm margin)
        """
        self.image_size = image_size
        self.margin = margin
        
        # Chọn reference landmarks
        if image_size == 112:
            self.ref_landmarks = self.REFERENCE_112.copy()
        elif image_size == 160:
            self.ref_landmarks = self.REFERENCE_160.copy()
        else:
            # Scale reference landmarks
            scale = image_size / 112.0
            self.ref_landmarks = self.REFERENCE_112 * scale
        
        # Thêm margin vào reference
        if margin > 0:
            center = np.mean(self.ref_landmarks, axis=0)
            self.ref_landmarks = self.ref_landmarks - center
            self.ref_landmarks = self.ref_landmarks * (1 + margin / image_size)
            self.ref_landmarks = self.ref_landmarks + center
    
    def align_landmarks(self, image, landmarks):
        """
        Align khuôn mặt dựa trên 5-point landmarks
        
        Args:
            image: numpy array (H, W, 3) - RGB image
            landmarks: numpy array (10,) - [x1, y1, x2, y2, ..., x5, y5]
        
        Returns:
            aligned_face: numpy array (image_size, image_size, 3) - Aligned face
        """
        if landmarks is None or len(landmarks) < 10:
            # Fallback: crop theo bounding box nếu không có landmarks
            return self._crop_without_landmarks(image, landmarks)
        
        # Reshape landmarks: (10,) -> (5, 2)
        src_landmarks = landmarks.reshape(5, 2).astype(np.float32)
        
        # Tính transformation matrix (similarity transform)
        if trans is not None:
            tform = trans.SimilarityTransform()
            tform.estimate(src_landmarks, self.ref_landmarks)
            M = tform.params[0:2, :]
        else:
            # Fallback: dùng cv2.getAffineTransform với 3 points
            # Chọn 3 landmarks: left eye, right eye, nose
            src_pts = src_landmarks[[0, 1, 2]].astype(np.float32)
            dst_pts = self.ref_landmarks[[0, 1, 2]].astype(np.float32)
            M = cv2.getAffineTransform(src_pts, dst_pts)  # Đã là 2x3 matrix
        
        # Warp ảnh
        aligned = cv2.warpAffine(
            image,
            M,
            (self.image_size, self.image_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        return aligned
    
    def align_bbox(self, image, bbox, landmarks=None):
        """
        Align khuôn mặt từ bounding box và landmarks (nếu có)
        
        Args:
            image: numpy array (H, W, 3) - RGB image
            bbox: numpy array (5,) - [x1, y1, x2, y2, confidence]
            landmarks: numpy array (10,) hoặc None
        
        Returns:
            aligned_face: numpy array (image_size, image_size, 3)
        """
        # Nếu có landmarks, dùng alignment
        if landmarks is not None and len(landmarks) >= 10:
            return self.align_landmarks(image, landmarks)
        
        # Nếu không có landmarks, crop theo bbox với margin
        return self._crop_bbox(image, bbox)
    
    def _crop_bbox(self, image, bbox):
        """Crop khuôn mặt từ bounding box với margin"""
        x1, y1, x2, y2 = bbox[0:4].astype(int)
        
        # Thêm margin
        h, w = image.shape[:2]
        margin_w = int((x2 - x1) * self.margin / 100.0) if self.margin > 0 else 0
        margin_h = int((y2 - y1) * self.margin / 100.0) if self.margin > 0 else 0
        
        x1 = max(0, x1 - margin_w)
        y1 = max(0, y1 - margin_h)
        x2 = min(w, x2 + margin_w)
        y2 = min(h, y2 + margin_h)
        
        # Crop
        cropped = image[y1:y2, x1:x2]
        
        # Resize về image_size
        if cropped.size > 0:
            aligned = cv2.resize(
                cropped,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_LINEAR
            )
        else:
            # Fallback: tạo ảnh đen
            aligned = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        return aligned
    
    def _crop_without_landmarks(self, image, landmarks):
        """Fallback crop khi không có landmarks"""
        # Tạo ảnh đen với kích thước chuẩn
        return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

