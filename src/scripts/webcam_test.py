import cv2
import argparse

parser = argparse.ArgumentParser(description="Hello")
parser.add_argument("-id", "--camera_id", type=int, default=0, help="ID of camera device. Run v4l2-ctl --list-devices to get more info")

args = parser.parse_args()

# Open the default webcam
cap = cv2.VideoCapture(args.camera_id)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # If the 'q' key is pressed, break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()