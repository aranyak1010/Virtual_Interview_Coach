# Install required libraries (Only needed for Google Colab)
import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Face Mesh model
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Function to analyze facial expressions & eye contact
def analyze_face():
    cap = cv2.VideoCapture(0)  # Open webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                continue

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process image
            results = face_mesh.process(rgb_frame)

            # Draw face mesh only if landmarks are detected
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        face_landmarks,
                        mp_face_mesh.FACEMESH_CONTOURS  # Removed landmark_drawing_spec
                    )

            # Display the output
            cv2.imshow('Facial Analysis', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()




# Run facial analysis
if __name__ == "__main__":
    analyze_face()
