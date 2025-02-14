import cv2
import os

# Set the folder where images will be saved
save_folder = "calibration_images"
os.makedirs(save_folder, exist_ok=True)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

img_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    # Show the live feed
    cv2.imshow("Webcam Feed", frame)
    
    # Listen for key events
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # Spacebar pressed
        img_name = os.path.join(save_folder, f"image_{img_count:04d}.jpg")
        cv2.imwrite(img_name, frame)
        print(f"Saved {img_name}")
        img_count += 1
    elif key == ord('q'):  # 'q' pressed
        break

# Release resources
cap.release()
cv2.destroyAllWindows()