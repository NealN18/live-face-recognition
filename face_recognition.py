import cv2
import os
from concurrent.futures import ThreadPoolExecutor
from face_detector import FaceDetector 
from face_processor import FaceProcessor
from result_renderer import ResultRenderer

class LiveFaceRecognition:
    def __init__(self, reference_dir="reference_images"):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.reference_dir = os.path.join(script_dir, reference_dir)
        self.face_results = []  
        self.processing = False
        self.new_results_ready = False 
        self.executor = ThreadPoolExecutor(max_workers=1)
        
        
        os.makedirs(os.path.expanduser("~/.deepface/weights"), exist_ok=True)
        
        
        if not os.path.exists(self.reference_dir):
            raise ValueError(f"Reference images folder '{self.reference_dir}' not found!")
        
        # Count reference images
        ref_files = [f for f in os.listdir(self.reference_dir) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not ref_files:
            raise ValueError(f"No valid reference images found in {self.reference_dir}")
        print(f"Loaded {len(ref_files)} reference images.")

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.face_detector = FaceDetector(self.reference_dir)
        self.face_processor = FaceProcessor(self.reference_dir)
        self.result_renderer = ResultRenderer()

        self._preload_model()

    def _preload_model(self):
        """Preload the face recognition model"""
        try:
            print("Preloading model...")
            sample_img_path = os.path.join(self.reference_dir, os.listdir(self.reference_dir)[0])
            self.face_processor.find_faces(
                img_path=sample_img_path,
                db_path=self.reference_dir
            )
            print("Model preloaded successfully.")
        except Exception as e:
            print(f"Error preloading model: {str(e)}")

    def check_face(self, frame):
        """Perform face verification for all detected faces"""
        try:
            self.processing = True
            self.new_results_ready = False
            
            self.face_results = self.face_detector.detect_and_recognize(frame, self.face_processor)
            self.new_results_ready = True

        except Exception as e:
            print(f"Face verification error: {str(e)}")
            pass
        finally:
            self.processing = False

    def run(self):
        """Main loop for face recognition"""
        counter = 0
        print("Starting video capture...")
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Process every 15 frames for better performance
                if counter % 15 == 0 and not self.processing:
                    self.executor.submit(self.check_face, frame.copy())
                counter += 1
                
                frame = self.result_renderer.draw_results(
                    frame, 
                    self.face_results, 
                    self.processing, 
                    self.new_results_ready
                )
                

                cv2.imshow('Multi-Face Recognition', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        except KeyboardInterrupt:
            print("Interrupted by user")
        except Exception as e:
            print(f"Error in main loop: {str(e)}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        self.executor.shutdown(wait=False)
        self.cap.release()
        cv2.destroyAllWindows()