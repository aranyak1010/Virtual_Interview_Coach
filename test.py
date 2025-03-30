import cv2

cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("Webcam is accessible!")
else:
    print("Webcam is NOT accessible. Check permissions.")
cap.release()
