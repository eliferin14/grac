import cv2
import argparse
import time

# Function to parse arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Test webcam and display FPS.")
    parser.add_argument("-id", "--camera_id", type=int, help="Camera device ID (e.g., 0 for default webcam, 1 for /dev/video1, etc.)")
    return parser.parse_args()

# Main function to test the webcam
def test_webcam(camera_id):
    # Open the camera
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"Error: Couldn't open camera with ID {camera_id}.")
        return

    # Variables for FPS calculation
    fps = 0
    last_time = time.time()

    # Capture frames in an infinite loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - last_time)  # FPS = 1 / time difference between frames
        last_time = current_time

        # Add FPS to the frame as red text
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('Webcam Test', frame)

        # Break the loop if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()

    # Test the webcam
    test_webcam(args.camera_id)