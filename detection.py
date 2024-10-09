import cv2
from ultralytics import YOLO
import yt_dlp

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
    if not cap.isOpened():
        print("Error: Could not open video stream")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        yield frame

    cap.release()

youtube_url = 'https://www.youtube.com/watch?v=KBsqQez-O4w'

# Extract the stream URL using yt-dlp
stream_url = get_youtube_stream_url(youtube_url)

if not stream_url:
    print("Error: Could not extract the stream URL.")
    exit()

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # You can use yolov8s.pt or yolov8m.pt based on your performance needs

# Define vehicle classes (car, truck, bus) using COCO class IDs
vehicle_classes = {2: 'car', 5: 'bus', 7: 'truck'}

# Read frames using OpenCV from the extracted stream URL
for frame in read_frame_from_stream(stream_url):

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Loop through the detections and filter for vehicles
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])  # Get the class ID
            if class_id in vehicle_classes:  # Check if the class ID is a vehicle (car, bus, truck)
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
                conf = box.conf[0]  # Get confidence score
                label = f"{vehicle_classes[class_id]} {conf:.2f}"  # Label with class name and confidence

                # Draw bounding boxes on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with bounding boxes
    cv2.imshow('YOLOv8 Vehicle Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
