from face_recognition import LiveFaceRecognition

def main():
    try:
        face_recognizer = LiveFaceRecognition()
        face_recognizer.run()
    except Exception as e:
        print(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()