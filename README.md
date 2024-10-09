# YouTube Object Detection with YOLOv5

This project is a Python-based implementation that uses **YOLOv5** to perform real-time object detection on cars in a live stream from YouTube. The YouTube video stream is processed using **yt-dlp** to retrieve the stream URL, and the video is analyzed frame by frame using **OpenCV**. YOLOv5 is used for object detection, specifically to detect cars, and bounding boxes are drawn around them in real-time.

## Features

- Extracts the best available video stream from YouTube using **yt-dlp**.
- Processes the video stream frame by frame using **OpenCV**.
- Detects objects in real-time using **YOLOv5**.
- Draws bounding boxes around detected cars in the video stream.

## Requirements

Before running this project, you need to have the following Python libraries installed:

- `yt-dlp` (for retrieving the YouTube video stream)
- `torch` (for using the YOLOv5 model)
- `opencv-python` (for processing the video stream and drawing bounding boxes)

You can install the required packages using `pip`:

```bash
pip install yt-dlp torch opencv-python
```

## How to Run

1. Clone or download this repository.
2. Make sure you have Python 3 installed on your machine.
3. Install the required libraries (see above).
4. Run the `detection.py` script:

   ```bash
   python3 detection.py
   ```

The script will automatically extract the best available YouTube video stream, process the frames, and use YOLOv5 to detect cars in real-time.

## Code Overview

### `get_youtube_stream_url(youtube_url)`

This function uses **yt-dlp** to extract the best video stream URL from a given YouTube video URL. It simplifies the process of getting the video stream, allowing it to be passed into OpenCV.

### `read_frame_from_stream(stream_url)`

This function opens the video stream using **OpenCV's `VideoCapture`** and reads frames from the stream in real-time, yielding each frame for processing.

### YOLOv5 Model Loading and Inference

YOLOv5 is loaded using **`torch.hub`**, and it performs object detection on each frame. The detections are filtered to only show cars, and bounding boxes are drawn around the detected cars.

## Example

Here’s an example of how you can replace the YouTube URL in the script with a different one:

```python
# Replace with your YouTube video URL
youtube_url = 'https://www.youtube.com/watch?v=YOUR_VIDEO_ID'
```

After running the script, you will see a video window where YOLOv5 detects and tracks cars, marking them with bounding boxes.

## Issues

If you encounter any issues, please ensure that:
- You have installed the required libraries.
- You are using a valid YouTube URL that provides a video stream.
- Your internet connection is stable enough to stream the video in real-time.

## License

This project is open-source and available under the MIT License.
