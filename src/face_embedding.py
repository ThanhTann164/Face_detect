"""
Face Embedding Module - Extract face embeddings
Đảm bảo preprocessing nhất quán giữa training và inference
"""
import numpy as np
import tensorflow as tf
import cv2
import sys
import os

# Import facenet
sys.path.insert(0, os.path.dirname(__file__))
import facenet

class FaceEmbedder:
    """
    Face embedder sử dụng FaceNet
    Đảm bảo preprocessing nhất quán
    """
    
    def __init__(self, model_path, image_size=160, gpu_memory_fraction=0.6):
        """
        Khởi tạo FaceNet embedder
        
        Args:
            model_path: Đường dẫn đến FaceNet .pb model
            image_size: Kích thước ảnh input (160 cho FaceNet)
            gpu_memory_fraction: Tỷ lệ GPU memory
        """
        self.model_path = model_path
        self.image_size = image_size
        self.gpu_memory_fraction = gpu_memory_fraction
        
        # Tạo graph và session
        self.graph = tf.Graph()
        self.sess = None
        self.images_placeholder = None
        self.embeddings = None
        self.phase_train_placeholder = None
        
        self._load_model()
    
    def _load_model(self):
        """Load FaceNet model"""
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
                # Load model
                facenet.load_model(self.model_path)
                
                # Get input and output tensors
                self.images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
                self.embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
                self.phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
                
                embedding_size = self.embeddings.get_shape()[1]
                print(f"[OK] FaceNet model loaded - embedding size: {embedding_size}")
    
    def preprocess(self, face_image, normalize=True):
        """
        Preprocess ảnh khuôn mặt - NHẤT QUÁN với training
        
        Args:
            face_image: numpy array (H, W, 3) - RGB image, uint8
            normalize: Có normalize (prewhiten) hay không
        
        Returns:
            processed: numpy array (1, image_size, image_size, 3) - float32
        """
        # Đảm bảo là RGB và uint8
        if face_image.dtype != np.uint8:
            face_image = (face_image * 255).astype(np.uint8) if face_image.max() <= 1.0 else face_image.astype(np.uint8)
        
        # Resize về image_size nếu cần
        if face_image.shape[0] != self.image_size or face_image.shape[1] != self.image_size:
            face_image = cv2.resize(
                face_image,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_LINEAR
            )
        
        # Prewhiten (normalize) - QUAN TRỌNG: phải giống training
        if normalize:
            face_image = facenet.prewhiten(face_image)
        else:
            # Chuyển sang float32 và normalize về [0, 1]
            face_image = face_image.astype(np.float32) / 255.0
        
        # Reshape thành batch: (1, H, W, 3)
        face_image = face_image.reshape(1, self.image_size, self.image_size, 3)
        
        return face_image
    
    def get_embedding(self, face_image, normalize=True):
        """
        Extract embedding từ ảnh khuôn mặt
        
        Args:
            face_image: numpy array (H, W, 3) - RGB image
            normalize: Có normalize embedding hay không (L2 normalize)
        
        Returns:
            embedding: numpy array (embedding_size,) - float32
        """
        # Preprocess
        processed = self.preprocess(face_image, normalize=True)  # Luôn normalize input
        
        # Extract embedding
        with self.graph.as_default():
            with self.sess.as_default():
                feed_dict = {
                    self.images_placeholder: processed,
                    self.phase_train_placeholder: False
                }
                embedding = self.sess.run(self.embeddings, feed_dict=feed_dict)
        
        # Lấy embedding đầu tiên (batch size = 1)
        embedding = embedding[0]
        
        # L2 normalize embedding (quan trọng cho cosine similarity)
        if normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        
        return embedding
    
    def get_embeddings_batch(self, face_images, normalize=True):
        """
        Extract embeddings từ batch ảnh
        
        Args:
            face_images: list of numpy arrays hoặc numpy array (N, H, W, 3)
            normalize: Có normalize embeddings hay không
        
        Returns:
            embeddings: numpy array (N, embedding_size)
        """
        if isinstance(face_images, list):
            # Preprocess từng ảnh
            processed_list = [self.preprocess(img, normalize=True) for img in face_images]
            processed = np.vstack(processed_list)
        else:
            # Đã là batch
            processed = face_images
        
        # Extract embeddings
        with self.graph.as_default():
            with self.sess.as_default():
                feed_dict = {
                    self.images_placeholder: processed,
                    self.phase_train_placeholder: False
                }
                embeddings = self.sess.run(self.embeddings, feed_dict=feed_dict)
        
        # L2 normalize
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)  # Tránh chia 0
            embeddings = embeddings / norms
        
        return embeddings
    
    def close(self):
        """Đóng session"""
        if self.sess is not None:
            self.sess.close()


