import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Launch scrcpy with specified camera id, video device, and resolution.")
    
    # Define arguments
    parser.add_argument("-id", "--camera-id", type=int, required=True, help="Camera ID to use (e.g., 2). 0=main, 1=front, 2=wide")
    parser.add_argument("-vd", "--video-device", type=int, required=True, help="Video device number (e.g., 3 for /dev/video3).")
    parser.add_argument("-r", "--resolution", type=int, required=True, help="Resolution for -m flag (e.g., 1280).")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Remove existing devices
    command = ["sudo", "modprobe", "-r", "v4l2loopback"]
    
    # Launch the command
    try:
        subprocess.run(command, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to execute command: {e}")
    
    # Build the command to create the device
    command = [
        "sudo", "modprobe", "v4l2loopback",
        f"video_nr={args.video_device}",
        "card_label=\"scrcpy stream\""
    ]
    
    # Launch the command
    try:
        subprocess.run(command, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to execute command: {e}")
    
    # List devices
    command = "v4l2-ctl --list-devices"
    
    # Launch the command
    try:
        subprocess.run(command, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to execute command: {e}")
    
    # Build the scrcpy command with the provided arguments
    command = [
        "scrcpy",
        "--video-source=camera",
        "--no-audio",
        "--no-window",
        f"--camera-id={args.camera_id}",
        f"--v4l2-sink=/dev/video{args.video_device}",
        f"-m{args.resolution}"
    ]
    
    # Launch the command
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to execute command: {e}")

if __name__ == "__main__":
    main()