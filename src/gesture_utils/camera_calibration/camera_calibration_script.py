import cv2
import numpy as np
import os
import glob

# Chessboard settings
CHESSBOARD_SIZE = (9,7)  # Adjust based on your pattern
SQUARE_SIZE = 0.02  # Set the actual size of a square in your units (e.g., cm)

# Termination criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (3D points in real-world space)
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE

# Lists to store object points and image points
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane

# Load images from folder
image_folder = "calibration_images"  # Change as needed
images = glob.glob(os.path.join(image_folder, "*.jpg"))

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
    
    if ret:
        # Refine corners
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners2)

# Perform camera calibration
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

if ret:
    # Save camera matrix and distortion coefficients
    np.savez("camera_calibration_results.npz", camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    print("Camera calibration successful. Data saved to camera_calibration_results.npz")
    print(f"{camera_matrix}")
    print(f"{dist_coeffs}")
    
    # Load one of the calibration images
    sample_img = cv2.imread(image_folder+"/image_0023.jpg")
    sample_gray = cv2.cvtColor(sample_img, cv2.COLOR_BGR2GRAY)
    
    # Undistort the image
    undistorted_img = cv2.undistort(sample_img, camera_matrix, dist_coeffs)
    ref_frame_image = sample_img.copy()
    
    ret, corners = cv2.findChessboardCorners(sample_gray, CHESSBOARD_SIZE, None)
    if ret:
        # Refine corners 
        corners2 = cv2.cornerSubPix(sample_gray, corners, (11, 11), (-1, -1), criteria)
        
        # Draw corners
        cv2.drawChessboardCorners(undistorted_img, CHESSBOARD_SIZE, corners2, ret)
        
        # Define reference frame axes 
        axis = np.float32([[0.04, 0, 0], [0, 0.04, 0], [0, 0, 0.04]]).reshape(-1, 3)
        
        # Solve PnP to get the pose
        _, rvec, tvec = cv2.solvePnP(objp, corners2, camera_matrix, dist_coeffs)
        
        # Project 3D points to image plane
        imgpts, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)
        
        origin = tuple(corners2[0].ravel().astype(int))
        imgpts = imgpts.astype(int)
        
        # Draw reference frame on undistorted image
        ref_frame_image = cv2.line(ref_frame_image, origin, tuple(imgpts[0].ravel()), (0, 0, 255), 3)  # X-axis (red)
        ref_frame_image = cv2.line(ref_frame_image, origin, tuple(imgpts[1].ravel()), (0, 255, 0), 3)  # Y-axis (green)
        ref_frame_image = cv2.line(ref_frame_image, origin, tuple(imgpts[2].ravel()), (255, 0, 0), 3)  # Z-axis (blue)
        
        # Concatenate original and undistorted images side by side
        comparison_img = np.hstack((sample_img, undistorted_img, ref_frame_image))
        
        # Show the images
        cv2.imshow('Original vs Undistorted with Reference Frame', comparison_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
else:
    print("Camera calibration failed.")
