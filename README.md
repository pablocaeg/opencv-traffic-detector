# 🚗 YOLO11 Vehicle Detection from YouTube Stream

This Python project uses **YOLO11** for real-time vehicle detection from a YouTube video stream. It tracks vehicles like **cars** and displays the total number of vehicles passed along with the elapsed time on the video stream.

## 📷 Demo

![Demo](./images/demo.gif)

## ✨ Features
- Real-time vehicle detection using **YOLO11**.
- Detects **cars** from YouTube live streams and videos.
- Tracks detected vehicles using **SORT** (Simple Online and Realtime Tracking).
- Displays bounding boxes and vehicle counts on the video stream.
- Shows **elapsed time** and **vehicle counts** on the video.

## 🛠️ Tech Stack
- **[Python](https://www.python.org/)** ![Python](https://img.shields.io/badge/Python-3.8-blue?logo=python)
- **[YOLO11](https://github.com/ultralytics/yolov5)** ![YOLO](https://img.shields.io/badge/YOLO-Object_Detection-orange)
- **[OpenCV](https://opencv.org/)** ![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-green)
- **[yt-dlp](https://github.com/yt-dlp/yt-dlp)** for extracting YouTube video streams
- **[SORT](https://github.com/abewley/sort)** for tracking objects
- **NumPy** ![NumPy](https://img.shields.io/badge/NumPy-Array_Processing-blue)

## 🚀 Quick Start

1. **Clone the repository**:

   ```bash
   git clone https://github.com/pablocaeg/opencv-traffic-detector.git
   cd opencv-traffic-detector
   ```

2. **Install dependencies**:

   ```bash
   pip install filterpy scikit-image numpy opencv-python torch ultralytics yt-dlp
   ```

3. **Install SORT**:

   SORT isn't available via `pip`, so you need to install it manually:

   ```bash
   git clone https://github.com/abewley/sort.git
   cd sort
   pip install -r requirements.txt
   python setup.py install
   ```

4. **Run the script**:

   ```bash
   python3 detection.py
   ```

5. **Replace the YouTube URL**:
   - In `detection.py`, replace the `youtube_url` variable with the YouTube video or livestream URL you want to process:

   ```python
   youtube_url = 'https://www.youtube.com/watch?v=YOUR_VIDEO_ID'
   ```

## 📊 Stats Displayed on Video
- **Total Cars Passed**: Counts and displays the total number of cars detected during the video stream.
- **Elapsed Time**: Shows how long the script has been running in **HH:MM:SS** format.

## 🖼️ Example Output

When the script is running, you'll see a video stream with **bounding boxes** around the detected cars and real-time stats overlaying the video.

Example of stats:
```
Cars Passed: 14
Time Elapsed: 00:12:43
```
