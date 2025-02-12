import os
import random
import shutil
import tensorflow as tf
assert tf.__version__.startswith('2')

from mediapipe_model_maker import gesture_recognizer

import argparse

import cv2
import numpy as np

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
# python3 dataset_builder.py -fdp /home/iris/Downloads/archive/hagrid-sample-30k-384p/hagrid_30k/ -mdp my_dataset --reduce_dataset -rdp reduced_dataset --gesture_whitelist fist palm one two three four pick L -mspg 100 --merge_palm_stop --merge_two_twoup

args = parser.parse_args()
print(args)

# Full and reduced dataset path
fdp = args.full_dataset_path
mdp = args.my_dataset_path
rdp = args.reduced_dataset_path
whitelist = args.gesture_whitelist + ['none']

# Calculate how many pictures to take from the full dataset and my dataset
target_num_fd = int( args.max_samples_per_gesture * args.full_my_ratio )
target_num_md = args.max_samples_per_gesture - target_num_fd
print(f"#images from full dataset: {target_num_fd}; #images from my dataset: {target_num_md}")

# Delete the old dataset and model
try:
    if args.reduce_dataset:
        shutil.rmtree(rdp)
        print(f"Folder '{rdp}' and its contents deleted successfully.")
        
except OSError as e:
    print(f"Error: {e.strerror}")




# Create the folder structure for the final dataset
for gesture in whitelist:
    rdp_gesture_path = os.path.join(rdp, gesture)
    if not os.path.exists(rdp_gesture_path):
        os.makedirs(rdp_gesture_path)

# Create the simplified full dataset -> all the gestures that are not in the whitelist are put into the 'none' class
sfdp = os.path.join(fdp, 'simplified_full_dataset')

# Remove the simplified datasets
shutil.rmtree(sfdp)
print(f"Folder '{sfdp}' and its contents deleted successfully.")

for gesture in whitelist:
    sfdp_gesture_path = os.path.join(sfdp, gesture)
    if not os.path.exists(sfdp_gesture_path):
        os.makedirs(sfdp_gesture_path)

# Scan the dataset and collapse it into the simplified dataset
for folder in os.listdir(fdp):
    # Check if the folder exists in the simplified dataset
    if folder == 'simplified_full_dataset':
        continue
    if folder in os.listdir(sfdp):
        target_folder = os.path.join(sfdp, folder)
    elif args.merge_palm_stop and (folder == 'stop' or folder == 'stop_inverted'):
        target_folder = os.path.join(sfdp, 'palm')
    elif args.merge_two_twoup and (folder == 'two_up' or folder == 'two_up_inverted') and 'two' in whitelist: 
        target_folder = os.path.join(sfdp, 'two')
    else:
        target_folder = os.path.join(sfdp, 'none')

    # Copy the images
    source_folder = os.path.join(fdp, folder)
    print(f"Copying images from {source_folder} to {target_folder} ...", end='')
    shutil.copytree(source_folder, target_folder, dirs_exist_ok=True)
    print("Done")






# Create the simplified my dataset -> all the gestures that are not in the whitelist are put into the 'none' class
smdp = os.path.join(mdp, 'simplified_my_dataset')

shutil.rmtree(smdp)
print(f"Folder '{smdp}' and its contents deleted successfully.")

for gesture in whitelist:
    smdp_gesture_path = os.path.join(smdp, gesture)
    if not os.path.exists(smdp_gesture_path):
        os.makedirs(smdp_gesture_path)

# Scan the dataset and collapse it into the simplified dataset
for folder in os.listdir(mdp):
    # Check if the folder exists in the simplified dataset
    if folder == 'simplified_my_dataset':
        continue
    if folder in os.listdir(smdp):
        target_folder = os.path.join(smdp, folder)
    else:
        target_folder = os.path.join(smdp, 'none')

    # Copy the images
    source_folder = os.path.join(mdp, folder)
    print(f"Copying images from {source_folder} to {target_folder} ...", end='')
    shutil.copytree(source_folder, target_folder, dirs_exist_ok=True)
    print("Done")



# Merge the two datasets into the final one
for gesture in whitelist:
    sfdp_source_folder = os.path.join(sfdp, gesture)
    smdp_source_folder = os.path.join(smdp, gesture)
    rdp_target_folder = os.path.join(rdp, gesture)
    assert os.path.exists(sfdp_source_folder) and os.path.exists(smdp_source_folder) and os.path.exists(rdp_target_folder)

    sfdp_files = os.listdir(sfdp_source_folder)
    smdp_files = os.listdir(smdp_source_folder)

    sfdp_image_count = len(sfdp_files)
    smdp_image_count = len(smdp_files)
    print(f"\'{gesture}\' images found in Full dataset: {sfdp_image_count}; in My dataset: {smdp_image_count}")

    # Decide how many images to take from full set (a) and my set (b)
    if sfdp_image_count >= target_num_fd and smdp_image_count >= target_num_md:
        a = target_num_fd
        b = target_num_md
    elif sfdp_image_count >= target_num_fd and smdp_image_count < target_num_md:
        b = smdp_image_count
        a = min( sfdp_image_count, args.max_samples_per_gesture - b)
    elif sfdp_image_count < target_num_fd and smdp_image_count >= target_num_md:
        a = sfdp_image_count
        b = min( smdp_image_count, args.max_samples_per_gesture - a)
    else:
        a = sfdp_image_count
        b = smdp_image_count
    print(f"A: {a}, B: {b} -> total images: {a+b}")

    # Shuffle to randomize the order
    random.shuffle(sfdp_files)
    random.shuffle(smdp_files)

    # Copy files from full dataset
    for i in range(a):
        src = os.path.join(sfdp_source_folder, sfdp_files[i])
        dst = os.path.join(rdp_target_folder, sfdp_files[i])
        print(f"{src}->{dst}")
        shutil.copy(src, dst)

    # Copy files from my dataset
    for i in range(b):
        src = os.path.join(smdp_source_folder, smdp_files[i])
        dst = os.path.join(rdp_target_folder, smdp_files[i])
        print(f"{src}->{dst}")
        shutil.copy(src, dst)





exit()

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
                
            # Check merge flags
            elif args.merge_palm_stop and (folder == 'stop' or folder == 'stop_inverted') and 'palm' in whitelist: 
                print(f"\tMerging \'{folder}\' into \'palm\'")          
                rdp_gesture_path = os.path.join(rdp, 'palm')
                
            elif args.merge_two_twoup and (folder == 'two_up' or folder == 'two_up_inverted') and 'two' in whitelist: 
                print(f"\tMerging \'{folder}\' into \'two\'")          
                rdp_gesture_path = os.path.join(rdp, 'two')
                
            else:
                print(f"\t{folder} -> blacklisted")            
                rdp_gesture_path = os.path.join(rdp, 'none')
                
            # Create the folder if necessary
            if not os.path.exists(rdp_gesture_path):
                os.makedirs(rdp_gesture_path)

            # Load all filenames and shuffle
            image_files = os.listdir(mdp_gesture_path)
            num_images = min(target_num_md, len(image_files)) if folder in os.listdir(rdp) else args.max_samples_per_gesture
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
            temp_num_fd = target_num_fd
            
        if os.path.isdir(fdp_gesture_path):
            
            # Check if the gesture is whitelisted
            if folder in whitelist:
                print(f"\t{folder} -> whitelisted")            
                rdp_gesture_path = os.path.join(rdp, folder)
                
            # Check merge flags
            elif args.merge_palm_stop and (folder == 'stop' or folder == 'stop_inverted') and 'palm' in whitelist: 
                print(f"\tMerging \'{folder}\' into \'palm\'")          
                rdp_gesture_path = os.path.join(rdp, 'palm')
                
            elif args.merge_two_twoup and (folder == 'two_up' or folder == 'two_up_inverted') and 'two' in whitelist: 
                print(f"\tMerging \'{folder}\' into \'two\'")          
                rdp_gesture_path = os.path.join(rdp, 'two')
                
            else:
                print(f"\t{folder} -> blacklisted")            
                rdp_gesture_path = os.path.join(rdp, 'none')
                
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