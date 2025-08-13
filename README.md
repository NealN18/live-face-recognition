# Multi-Face Recognition with DeepFace

This project performs real-time face recognition on multiple faces using a webcam. It leverages the DeepFace library for face recognition and employs a multi-threaded approach to separate the potentially slow recognition process from the main video capture loop, aiming for smoother video display.

## Features

*  **Real-time Multi-Face Recognition:** Processes webcam feed in real-time and detects multiple faces.
*  **Face Recognition:** Uses DeepFace with the Facenet model to recognize faces against a database of reference images.
*  **Multi-Threading:** Offloads the DeepFace recognition to a separate thread using Python's concurrent.futures.ThreadPoolExecutor. This keeps the main OpenCV loop responsive for video display.
*  **Persistent Results:** Displays the last known recognition results while new frames are being processed.
*  **Performance Considerations:** Processes every Nth frame (configurable) to manage performance.

## How It Works

1. The main thread initializes the webcam and loads reference face images from a specified directory.
2. The recognition model (Facenet) is preloaded using a sample reference image.
3. The main thread captures video frames from the default webcam.
4. Based on a frame counter, the main thread decides whether to trigger a new recognition analysis.
5. If triggered, the main thread submits the current frame to a thread pool executor for processing by the check_face function.
6. The check_face function (running in a separate thread) detects all faces in the frame using RetinaFace.
7. For each detected face, it extracts the face region and performs recognition against the reference database using DeepFace.find.
8. The recognition results (bounding box, match status, distance) for all detected faces are stored in a shared list (face_results).
9. The main thread continuously draws the latest available results (from face_results) onto the current frame and displays it.
10. This process repeats, updating the results every few frames.

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/NealN18/age-gender-detection.git
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: DeepFace will automatically download the required models (e.g., for RetinaFace detection, Facenet recognition) on the first run. These are typically stored in ~/.deepface/weights/.*

3. **Prepare Reference Images:**
Prepare Reference Images:
*  Create a folder named reference_images in the project directory.
*  Add clear, front/side-facing images of the people you want to recognize to this folder. Supported formats are .png, .jpg, .jpeg.
  
## Usage
1. Ensure your webcam is connected and accessible.
2. Ensure you have prepared your reference_images folder with at least one image(although the more images the better).

3. A window will open showing the webcam feed.
4. Detected faces will be highlighted with bounding boxes:
* Green Box + "Match!": A face is recognized as one from the reference_images folder.
* Red Box + "No Match": A face is detected but not recognized.
5. The bottom of the screen shows the number of faces currently detected/recognized.
6. Press q to quit the application.

## Project Structure
*  **main.py:** Entry point of the application.
*  **face_recognition_app.py:** Core application logic, including camera handling and threading setup.
*  **face_detector.py:** Handles face detection (using RetinaFace) and initiates recognition for each detected face.
*  **face_processor.py:** Wrapper for the DeepFace recognition logic.
*  **result_renderer.py:** Handles drawing bounding boxes, labels, and status information onto the video frame.
*  **requirements.txt:** List of required Python packages.

## Dependencies
*See requirements.txt for the list of required Python packages.*

## Acknowledgements
*   [DeepFace](https://github.com/serengil/deepface)
*   [OpenCV](https://opencv.org/)
*   [RetinaFace](https://github.com/serengil/retinaface)
