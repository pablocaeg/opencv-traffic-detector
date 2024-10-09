import cv2
from ultralytics import YOLO
import yt_dlp
import numpy as np
from sort import Sort
import time  # For tracking elapsed time

# Function to extract the direct video stream URL from YouTube using yt-dlp
def get_youtube_stream_url(youtube_url):
    ydl_opts = {
        'format': 'best',  # Get the best available video+audio format
        'noplaylist': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info['url']  # Return the best stream URL

# Function to read frames from the stream using OpenCV
def read_frame_from_stream(stream_url):
    cap = cv2.VideoCapture(stream_url)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get frames per second
    if not cap.isOpened():
        print("Error: Could not open video stream")
        return None, fps

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame or the video ended")
            break

        yield frame, fps

    cap.release()

# Replace with your YouTube video URL
youtube_url = 'https://www.youtube.com/watch?v=KBsqQez-O4w'

# Extract the stream URL using yt-dlp
stream_url = get_youtube_stream_url(youtube_url)

if not stream_url:
    print("Error: Could not extract the stream URL.")
    exit()

# Load the YOLO model
model = YOLO('yolo11n.pt')  # Use the nano model for faster inference

# Define vehicle class (car) using COCO class ID
car_class_id = 2  # Class ID 2 corresponds to 'car'

# Initialize the SORT tracker
tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.1)

# Desired target resolution for better performance
target_width = 640
target_height = 360

# Initialize variables to track stats
start_time = time.time()  # Store the time the script started
last_object_id = 0  # Last assigned object ID

# Read frames using OpenCV from the extracted stream URL
for frame, fps in read_frame_from_stream(stream_url):

    # Get original frame dimensions
    original_height, original_width = frame.shape[:2]

    # Resize frame for faster detection
    frame_resized = cv2.resize(frame, (target_width, target_height))

    # Run YOLO inference on the resized frame
    results = model(frame_resized)

    # Prepare detections for the tracker (rescale to original size for bounding boxes)
    detections = []
    for result in results:
        boxes = result.boxes
        for i, box in enumerate(boxes):
            class_id = int(box.cls[0])  # Get the class ID
            if class_id == car_class_id:  # Check if the class ID is 'car'
                # Get bounding box coordinates and confidence score (rescale to original size)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Scale the coordinates back to the original frame size
                x1 = int(x1 * original_width / target_width)
                y1 = int(y1 * original_height / target_height)
                x2 = int(x2 * original_width / target_width)
                y2 = int(y2 * original_height / target_height)

                conf = box.conf[0]  # Get confidence score
                detections.append([x1, y1, x2, y2, conf])

    # Update tracker with new detections
    if detections:  # Ensure there are valid detections
        tracked_objects = tracker.update(np.array(detections))

        # Loop through tracked objects and draw bounding boxes
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = map(int, obj[:5])  # Get the bounding box and object ID

            # Update last_object_id to the current object's ID
            if obj_id > last_object_id:
                last_object_id = obj_id

            # Draw bounding boxes on the original frame
            label = f"car {obj_id}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Calculate time elapsed since the start of the script
    elapsed_time = time.time() - start_time
    elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

    # Display stats on the screen (total cars passed and elapsed time)
    cv2.putText(frame, f"Cars Passed: {last_object_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Time Elapsed: {elapsed_time_str}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display the original frame with bounding boxes and stats
    cv2.imshow('YOLO Car Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
