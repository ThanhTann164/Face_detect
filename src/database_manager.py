"""
Database Manager - Quản lý embeddings database
Lưu và load embeddings dạng JSON, hỗ trợ thêm/xóa người mới nhanh chóng
"""
import os
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pickle

class EmbeddingsDatabase:
    """
    Quản lý database embeddings cho face recognition
    Format: {
        "version": "1.0",
        "model_info": {
            "facenet_model": "...",
            "image_size": 160,
            "embedding_size": 512
        },
        "persons": {
            "person_name": {
                "embeddings": [[...], [...], ...],  # List of embeddings
                "created_at": "2025-11-23T16:00:00",
                "updated_at": "2025-11-23T16:00:00",
                "num_images": 10
            }
        }
    }
    """
    
    def __init__(self, db_path: str):
        """
        Khởi tạo database manager
        
        Args:
            db_path: Đường dẫn file database (.json hoặc .pkl)
        """
        self.db_path = db_path
        self.db_data = None
        self._load_database()
    
    def _load_database(self):
        """Load database từ file"""
        if os.path.exists(self.db_path):
            try:
                if self.db_path.endswith('.json'):
                    with open(self.db_path, 'r', encoding='utf-8') as f:
                        self.db_data = json.load(f)
                elif self.db_path.endswith('.pkl'):
                    with open(self.db_path, 'rb') as f:
                        self.db_data = pickle.load(f)
                else:
                    raise ValueError(f"Unsupported database format: {self.db_path}")
                
                # Convert embeddings từ list sang numpy array
                self._convert_embeddings_to_numpy()
                
                print(f"[OK] Loaded database: {len(self.db_data.get('persons', {}))} persons")
            except Exception as e:
                print(f"[ERROR] Failed to load database: {e}")
                self.db_data = self._create_empty_database()
        else:
            self.db_data = self._create_empty_database()
            self._save_database()
    
    def _create_empty_database(self) -> Dict:
        """Tạo database rỗng"""
        return {
            "version": "1.0",
            "model_info": {
                "facenet_model": "",
                "image_size": 160,
                "embedding_size": 512
            },
            "persons": {}
        }
    
    def _convert_embeddings_to_numpy(self):
        """Convert embeddings từ list sang numpy array"""
        for person_name, person_data in self.db_data.get('persons', {}).items():
            if 'embeddings' in person_data:
                if isinstance(person_data['embeddings'], list):
                    person_data['embeddings'] = np.array(person_data['embeddings'], dtype=np.float32)
    
    def _save_database(self):
        """Lưu database vào file"""
        try:
            # Convert numpy arrays to list trước khi save JSON
            db_to_save = json.loads(json.dumps(self.db_data, default=self._numpy_serializer))
            
            if self.db_path.endswith('.json'):
                with open(self.db_path, 'w', encoding='utf-8') as f:
                    json.dump(db_to_save, f, indent=2, ensure_ascii=False)
            elif self.db_path.endswith('.pkl'):
                with open(self.db_path, 'wb') as f:
                    pickle.dump(self.db_data, f)
            
            print(f"[OK] Saved database to {self.db_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save database: {e}")
            raise
    
    def _numpy_serializer(self, obj):
        """Helper để serialize numpy arrays"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        raise TypeError(f"Type {type(obj)} not serializable")
    
    def set_model_info(self, facenet_model: str, image_size: int = 160, embedding_size: int = 512):
        """Set thông tin model"""
        self.db_data['model_info'] = {
            "facenet_model": facenet_model,
            "image_size": image_size,
            "embedding_size": embedding_size
        }
        self._save_database()
    
    def add_person(self, person_name: str, embeddings: np.ndarray, replace: bool = False):
        """
        Thêm embeddings cho một người
        
        Args:
            person_name: Tên người (chữ thường, không dấu)
            embeddings: numpy array (N, embedding_size) - embeddings của N ảnh
            replace: Nếu True, thay thế embeddings cũ; nếu False, append vào
        """
        person_name = person_name.lower().strip()
        
        if person_name not in self.db_data['persons']:
            self.db_data['persons'][person_name] = {
                "embeddings": [],
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "num_images": 0
            }
        
        # Convert embeddings sang numpy nếu cần
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings, dtype=np.float32)
        
        # Ensure 2D array
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        
        if replace:
            self.db_data['persons'][person_name]['embeddings'] = embeddings
        else:
            # Append embeddings mới vào embeddings cũ
            old_embeddings = self.db_data['persons'][person_name]['embeddings']
            if isinstance(old_embeddings, list):
                old_embeddings = np.array(old_embeddings, dtype=np.float32)
            if old_embeddings.size > 0:
                self.db_data['persons'][person_name]['embeddings'] = np.vstack([old_embeddings, embeddings])
            else:
                self.db_data['persons'][person_name]['embeddings'] = embeddings
        
        # Update metadata
        self.db_data['persons'][person_name]['updated_at'] = datetime.now().isoformat()
        self.db_data['persons'][person_name]['num_images'] = len(self.db_data['persons'][person_name]['embeddings'])
        
        self._save_database()
        print(f"[OK] Added {len(embeddings)} embeddings for person: {person_name} (Total: {self.db_data['persons'][person_name]['num_images']})")
    
    def remove_person(self, person_name: str) -> bool:
        """Xóa một người khỏi database"""
        person_name = person_name.lower().strip()
        if person_name in self.db_data['persons']:
            del self.db_data['persons'][person_name]
            self._save_database()
            print(f"[OK] Removed person: {person_name}")
            return True
        return False
    
    def get_person_embeddings(self, person_name: str) -> Optional[np.ndarray]:
        """Lấy embeddings của một người"""
        person_name = person_name.lower().strip()
        if person_name in self.db_data['persons']:
            embeddings = self.db_data['persons'][person_name]['embeddings']
            if isinstance(embeddings, list):
                embeddings = np.array(embeddings, dtype=np.float32)
            return embeddings
        return None
    
    def get_all_embeddings(self) -> Tuple[np.ndarray, List[str]]:
        """
        Lấy tất cả embeddings và labels
        
        Returns:
            embeddings: numpy array (N, embedding_size) - Tất cả embeddings
            labels: List[str] - Tên người tương ứng với mỗi embedding
        """
        all_embeddings = []
        all_labels = []
        
        for person_name, person_data in self.db_data['persons'].items():
            embeddings = person_data['embeddings']
            if isinstance(embeddings, list):
                embeddings = np.array(embeddings, dtype=np.float32)
            
            if embeddings.size > 0:
                # Ensure 2D
                if embeddings.ndim == 1:
                    embeddings = embeddings.reshape(1, -1)
                
                all_embeddings.append(embeddings)
                # Tạo labels cho mỗi embedding
                num_embeddings = len(embeddings) if embeddings.ndim == 2 else 1
                all_labels.extend([person_name] * num_embeddings)
        
        if len(all_embeddings) == 0:
            return np.array([]), []
        
        # Stack tất cả embeddings
        all_embeddings = np.vstack(all_embeddings)
        
        return all_embeddings, all_labels
    
    def get_all_persons(self) -> List[str]:
        """Lấy danh sách tất cả người trong database"""
        return list(self.db_data['persons'].keys())
    
    def get_person_info(self, person_name: str) -> Optional[Dict]:
        """Lấy thông tin của một người"""
        person_name = person_name.lower().strip()
        if person_name in self.db_data['persons']:
            info = self.db_data['persons'][person_name].copy()
            # Convert embeddings to list để JSON serializable
            if isinstance(info['embeddings'], np.ndarray):
                info['embeddings'] = info['embeddings'].tolist()
            return info
        return None
    
    def get_statistics(self) -> Dict:
        """Lấy thống kê database"""
        total_persons = len(self.db_data['persons'])
        total_embeddings = sum(
            person_data['num_images'] 
            for person_data in self.db_data['persons'].values()
        )
        
        return {
            "total_persons": total_persons,
            "total_embeddings": total_embeddings,
            "model_info": self.db_data['model_info'],
            "persons": {
                name: {
                    "num_images": data['num_images'],
                    "created_at": data['created_at'],
                    "updated_at": data['updated_at']
                }
                for name, data in self.db_data['persons'].items()
            }
        }
    
    def find_best_match(self, query_embedding: np.ndarray, threshold: float = 0.5) -> Tuple[Optional[str], float, bool]:
        """
        Tìm người khớp nhất với embedding query
        
        Args:
            query_embedding: numpy array (embedding_size,) - Embedding cần tìm
            threshold: Ngưỡng cosine similarity (0.0 - 1.0)
        
        Returns:
            best_person: Tên người khớp nhất hoặc None
            best_score: Cosine similarity score (0.0 - 1.0)
            is_match: True nếu best_score >= threshold, False nếu không
        """
        if len(self.db_data['persons']) == 0:
            return None, 0.0, False
        
        # Ensure query embedding là 1D và normalized
        if query_embedding.ndim > 1:
            query_embedding = query_embedding.flatten()
        
        # Normalize query embedding
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm
        
        best_person = None
        best_score = 0.0
        
        # So sánh với mỗi người
        for person_name, person_data in self.db_data['persons'].items():
            embeddings = person_data['embeddings']
            if isinstance(embeddings, list):
                embeddings = np.array(embeddings, dtype=np.float32)
            
            if embeddings.size == 0:
                continue
            
            # Ensure 2D
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            
            # Normalize embeddings
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            normalized_embeddings = embeddings / norms
            
            # Tính cosine similarity
            similarities = np.dot(normalized_embeddings, query_embedding)
            
            # Lấy similarity cao nhất của người này
            max_similarity = float(np.max(similarities))
            
            if max_similarity > best_score:
                best_score = max_similarity
                best_person = person_name
        
        # Kiểm tra threshold
        is_match = best_score >= threshold
        
        return best_person, best_score, is_match

