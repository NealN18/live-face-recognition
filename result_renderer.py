import cv2

class ResultRenderer:
    def draw_results(self, frame, face_results, processing, new_results_ready):
        """Draw results on frame for multiple faces"""
        for face_result in face_results:
            x, y, w, h = face_result["bbox"]
            if face_result["match"]:
                color = (0, 255, 0)  
                label = f"Match! ({face_result['distance']:.2f})"
            else:
                color = (0, 0, 255)  
                label = "No Match"
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label_width = len(label) * 12
            cv2.rectangle(frame, (x, y - 30), (x + label_width, y), color, -1)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if processing and not new_results_ready: 
            status_text = "PROCESSING..."
            status_color = (255, 255, 0)  
        else:
            status_text = f"FACES: {len(face_results)}"
            status_color = (255, 255, 255)  
            
        cv2.putText(frame, status_text, (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        cv2.putText(frame, "Press 'q' to quit", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame