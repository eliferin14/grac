import os
import random
import shutil
import tensorflow as tf
assert tf.__version__.startswith('2')

from mediapipe_model_maker import gesture_recognizer

import argparse

import cv2
import numpy as np

def rotate_image(img, angle):
    """Rotates an image by a given angle.

    Args:
        img: The input image as a NumPy array.
        angle: The rotation angle in degrees.

    Returns:
        The rotated image.
    """

    height, width = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
    rotated_image = cv2.warpAffine(img, rotation_matrix, (width, height))
    return rotated_image   

parser = argparse.ArgumentParser(description="Hello")
parser.add_argument("-fdp", "--full_dataset_path", type=str, default='dataset', help="ID of camera device. Run v4l2-ctl --list-devices to get more info")
parser.add_argument("-mdp", "--my_dataset_path", type=str, default='dataset', help="ID of camera device. Run v4l2-ctl --list-devices to get more info")
parser.add_argument("-fmr", "--full_my_ratio", type=float, default=0.5, help="Fraction of the dataset coming from my dataset")
parser.add_argument("--reduce_dataset", action='store_true', help="If True a new dataset is built")
parser.add_argument("-rdp", "--reduced_dataset_path", type=str, default='dataset', help="ID of camera device. Run v4l2-ctl --list-devices to get more info")
parser.add_argument('--gesture_whitelist', nargs='+', default=['none'], help='Whitelist of gestures. \'none\' is included by default')
parser.add_argument('-mspg', '--max_samples_per_gesture', type=int, default=200, help='Max number of samples per image')
parser.add_argument("--merge_palm_stop", action='store_true', help="If True palm and stop are considered both palm")
parser.add_argument("--merge_two_twoup", action='store_true', help="If True two and two_up are considered both two")
parser.add_argument("--add_random_rotation", action='store_true', help="If True the images are rotated by a random angle when copied")

args = parser.parse_args()
print(args)

# Full and reduced dataset path
fdp = args.full_dataset_path
mdp = args.my_dataset_path
rdp = args.reduced_dataset_path
whitelist = args.gesture_whitelist

# Calculate how many pictures to take from the full dataset and my dataset
num_fd = int( args.max_samples_per_gesture * args.full_my_ratio )
num_md = args.max_samples_per_gesture - num_fd

# Delete the old dataset and model
try:
    if args.reduce_dataset:
        shutil.rmtree(rdp)
        print(f"Folder '{rdp}' and its contents deleted successfully.")
except OSError as e:
    print(f"Error: {e.strerror}")

# Scan the full dataset, and copy the whitelisted gestures in their folder, and the blacklisted gestures in the none folder
if args.reduce_dataset:
                
                
    print("\nGestures found in my dataset:")
    for folder in os.listdir(mdp):
        
        mdp_gesture_path = os.path.join(mdp, folder)
        if os.path.isdir(mdp_gesture_path):
            
            # Check if the gesture is whitelisted
            if folder in whitelist:
                print(f"\t{folder} -> whitelisted")            
                rdp_gesture_path = os.path.join(rdp, folder)
                
            else:
                print(f"\t{folder} -> blacklisted")            
                rdp_gesture_path = os.path.join(rdp, 'none')
                
            # Check merge flags
            if args.merge_palm_stop and folder == 'stop' and 'palm' in whitelist: 
                print("\tMerging \'stop\' into \'palm\'")          
                rdp_gesture_path = os.path.join(rdp, 'palm')
                
            if args.merge_two_twoup and folder == 'two_up' and 'two' in whitelist: 
                print("\tMerging \'two_up\' into \'two\'")          
                rdp_gesture_path = os.path.join(rdp, 'two')
                
            # Create the folder if necessary
            if not os.path.exists(rdp_gesture_path):
                os.makedirs(rdp_gesture_path)

            # Load all filenames and shuffle
            image_files = os.listdir(mdp_gesture_path)
            num_images = min(num_md, len(image_files))
            random.shuffle(image_files)

            # Copy the desired number of files
            for i in range(num_images):
                source_file = os.path.join(mdp_gesture_path, image_files[i])
                dest_file = os.path.join(rdp_gesture_path, image_files[i])
                
                image = cv2.imread(source_file)
                
                if args.add_random_rotation:
                    
                    coinflip = random.uniform(-1,1)
                    if coinflip > 0:
                
                        # Generate a random angle between -90 and 90 degrees
                        random_angle = random.uniform(-90, 90)

                        # Rotate the image
                        image = rotate_image(image, random_angle)
                
                cv2.imwrite(dest_file, image)
                
                
                
    
    print("\nGestures found in the full dataset:")
    for folder in os.listdir(fdp):
        
        fdp_gesture_path = os.path.join(fdp, folder)
        
        if folder not in os.listdir(rdp):
            temp_num_fd = args.max_samples_per_gesture    
        else:        
            temp_num_fd = num_fd
            
        if os.path.isdir(fdp_gesture_path):
            
            # Check if the gesture is whitelisted
            if folder in whitelist:
                print(f"\t{folder} -> whitelisted")            
                rdp_gesture_path = os.path.join(rdp, folder)
                
            else:
                print(f"\t{folder} -> blacklisted")            
                rdp_gesture_path = os.path.join(rdp, 'none')
                
            # Check merge flags
            if args.merge_palm_stop and folder == 'stop' and 'palm' in whitelist: 
                print("\tMerging \'stop\' into \'palm\'")          
                rdp_gesture_path = os.path.join(rdp, 'palm')
                
            if args.merge_two_twoup and folder == 'two_up' and 'two' in whitelist: 
                print("\tMerging \'two_up\' into \'two\'")          
                rdp_gesture_path = os.path.join(rdp, 'two')
                
            # Create the folder if necessary
            if not os.path.exists(rdp_gesture_path):
                os.makedirs(rdp_gesture_path)

            # Load all filenames and shuffle
            image_files = os.listdir(fdp_gesture_path)
            num_images = min(temp_num_fd, len(image_files))
            random.shuffle(image_files)

            # Copy the desired number of files
            for i in range(num_images):
                source_file = os.path.join(fdp_gesture_path, image_files[i])
                dest_file = os.path.join(rdp_gesture_path, image_files[i])
                
                image = cv2.imread(source_file)
                
                if args.add_random_rotation:
                    
                    coinflip = random.uniform(-1,1)
                    if coinflip > 0:
                
                        # Generate a random angle between -90 and 90 degrees
                        random_angle = random.uniform(-90, 90)

                        # Rotate the image
                        image = rotate_image(image, random_angle)
                
                cv2.imwrite(dest_file, image)