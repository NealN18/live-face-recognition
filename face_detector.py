import cv2
import numpy as np
from retinaface import RetinaFace

class FaceDetector:
    def __init__(self, reference_dir):
        self.reference_dir = reference_dir

    def detect_and_recognize(self, frame, face_processor):
        """Detect faces and perform recognition"""
        face_results = []
        
        try:
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            
            resp = RetinaFace.detect_faces(small_frame)
            
            if isinstance(resp, dict):
                # Handle single face or multiple faces
                face_keys = [key for key in resp.keys() if key.startswith("face_")]
                if not face_keys and "face_1" not in resp:

                    face_keys = ["face_1"] if "score" in resp else []
                
                for face_key in face_keys:
                    if face_key in resp:
                        face_data = resp[face_key]
                    elif face_key == "face_1" and "score" in resp:
                        face_data = resp
                    else:
                        continue
                    
                    facial_area = face_data["facial_area"]
                    x, y, x2, y2 = facial_area
                    w, h = x2 - x, y2 - y
                    
                    # Scale coordinates back to original frame size
                    x, y, w, h = int(x * 2), int(y * 2), int(w * 2), int(h * 2)
                    
                    y1_adj = max(0, y // 2)
                    y2_adj = min(small_frame.shape[0], (y + h) // 2)
                    x1_adj = max(0, x // 2)
                    x2_adj = min(small_frame.shape[1], (x + w) // 2)
                    face_img = small_frame[y1_adj:y2_adj, x1_adj:x2_adj]
                    
                    if face_img.size > 0:
                        result = face_processor.find_faces(
                            img_path=face_img,
                            db_path=self.reference_dir
                        )
                        
                        if result is not None and not result.empty and 'distance' in result.columns and len(result) > 0:
                            match_distance = float(result.iloc[0]["distance"])
                            is_match = match_distance < 0.5
                            face_results.append({
                                "bbox": (x, y, w, h),
                                "match": is_match,
                                "distance": match_distance
                            })
                        else:
                            face_results.append({
                                "bbox": (x, y, w, h),
                                "match": False,
                                "distance": 1.0
                            })
        except Exception as detect_error:
            print(f"Face detection error: {str(detect_error)}")
            # Fallback method
            face_results = self._simple_detection(frame, face_processor)
            
        return face_results

    def _simple_detection(self, frame, face_processor):
        """Fallback method for face detection"""
        try:

            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            
            result = face_processor.find_faces(
                img_path=small_frame,
                db_path=self.reference_dir
            )
            
            return []
        except Exception as e:
            print(f"Fallback detection error: {str(e)}")
            return []