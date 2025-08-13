from deepface import DeepFace
import pandas as pd

class FaceProcessor:
    def __init__(self, reference_dir):
        self.reference_dir = reference_dir

    def find_faces(self, img_path, db_path):
        """Find and recognize faces using DeepFace"""
        try:
            results = DeepFace.find(
                img_path=img_path,
                db_path=db_path,
                model_name="Facenet",
                detector_backend="retinaface",
                enforce_detection=False
            )
            
            if isinstance(results, list):
                if len(results) > 0 and len(results[0]) > 0:
                    return results[0]
                else:
                    return pd.DataFrame()
            else:
                return results
        except Exception as e:
            print(f"Face processing error: {str(e)}")
            return None