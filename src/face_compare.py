"""
Face Comparison Module - So sánh embeddings
Sử dụng cosine similarity và distance threshold
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

class FaceComparator:
    """
    Face comparator sử dụng cosine similarity và euclidean distance
    """
    
    def __init__(self, method='cosine', threshold=None):
        """
        Khởi tạo comparator
        
        Args:
            method: 'cosine' hoặc 'euclidean'
            threshold: Ngưỡng để xác định match
                     - cosine: > threshold là match (default: 0.6)
                     - euclidean: < threshold là match (default: 1.2)
        """
        self.method = method.lower()
        
        if threshold is None:
            if self.method == 'cosine':
                self.threshold = 0.6  # Cosine similarity > 0.6
            else:
                self.threshold = 1.2  # Euclidean distance < 1.2
        else:
            self.threshold = threshold
    
    def compare(self, embedding1, embedding2):
        """
        So sánh 2 embeddings
        
        Args:
            embedding1: numpy array (embedding_size,)
            embedding2: numpy array (embedding_size,)
        
        Returns:
            similarity: float - similarity score
            is_match: bool - có match hay không
        """
        # Reshape thành (1, embedding_size) để dùng với sklearn
        emb1 = embedding1.reshape(1, -1)
        emb2 = embedding2.reshape(1, -1)
        
        if self.method == 'cosine':
            similarity = cosine_similarity(emb1, emb2)[0, 0]
            is_match = similarity > self.threshold
        else:  # euclidean
            distance = euclidean_distances(emb1, emb2)[0, 0]
            similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity
            is_match = distance < self.threshold
        
        return similarity, is_match
    
    def find_best_match(self, query_embedding, database_embeddings, database_labels):
        """
        Tìm match tốt nhất trong database
        
        Args:
            query_embedding: numpy array (embedding_size,)
            database_embeddings: numpy array (N, embedding_size)
            database_labels: list hoặc array (N,) - labels tương ứng
        
        Returns:
            best_label: str - label của match tốt nhất
            best_similarity: float - similarity score
            is_match: bool - có match hay không
        """
        if len(database_embeddings) == 0:
            return None, 0.0, False
        
        # Reshape query embedding
        query = query_embedding.reshape(1, -1)
        
        if self.method == 'cosine':
            # Cosine similarity
            similarities = cosine_similarity(query, database_embeddings)[0]
            best_idx = np.argmax(similarities)
            best_similarity = similarities[best_idx]
            is_match = best_similarity > self.threshold
        else:
            # Euclidean distance
            distances = euclidean_distances(query, database_embeddings)[0]
            best_idx = np.argmin(distances)
            best_distance = distances[best_idx]
            best_similarity = 1.0 / (1.0 + best_distance)
            is_match = best_distance < self.threshold
        
        best_label = database_labels[best_idx] if database_labels is not None else None
        
        return best_label, best_similarity, is_match
    
    def set_threshold(self, threshold):
        """Cập nhật threshold"""
        self.threshold = threshold
        print(f"[INFO] Updated {self.method} threshold to {threshold}")

