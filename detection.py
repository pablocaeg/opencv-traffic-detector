import cv2
import torch
import yt_dlp

# Function to extract the direct video stream URL from YouTube using yt-dlp
def get_youtube_stream_url(youtube_url):
    ydl_opts = {
        'format': 'best',
        'noplaylist': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info['url']

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

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Read frames using OpenCV from the extracted stream URL
for frame in read_frame_from_stream(stream_url):

    # Run YOLOv5 inference on the frame
    results = model(frame)

    # Filter for car detections
    results = results.pandas().xyxy[0]
    car_detections = results[results['name'] == 'car']

    # Draw bounding boxes on the frame for cars
    for _, row in car_detections.iterrows():
        x1, y1, x2, y2, conf = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), row['confidence']
        label = f"car {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with the car bounding boxes
    cv2.imshow('YOLOv5 Car Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
