# SENTINEL 👁️
**AI-Based Multi-Camera Person Re-Identification & Surveillance Dashboard**

SENTINEL is an end-to-end, full-stack application that detects, tracks, and matches identities across multiple independent camera streams. It combines a heavy-duty Python AI pipeline (YOLOv8 + DeepSORT + OSNet) with a lightning-fast modern web dashboard (Node.js + React/Vite) for seamless video uploading, processing, and playback.

---

## ✨ Key Features

* **Multi-Camera Re-ID Pipeline:** Matches human identities across completely different video feeds.
* **Full-Stack Web Dashboard:** A sleek, dark-themed React UI to submit jobs, monitor real-time processing logs, and stream the final output.
* **Node.js Orchestration:** An Express backend that acts as the task master, dynamically spawning and monitoring the Python AI processes.
* **Web-Ready Video Encoding:** Automatically processes and encodes OpenCV outputs into H.264 (`avc1`) for native HTML5 browser playback.
* **Cloud Telemetry:** Integrates directly with Firebase Firestore to log run summaries, detection counts, and processing times.

## 🛠️ Tech Stack

* **AI & Computer Vision:** Python 3.10+, OpenCV, YOLOv8s (Ultralytics), DeepSORT, Torchreid (OSNet).
* **Backend:** Node.js, Express, Multer (File Handling).
* **Frontend:** React, Vite, Tailwind CSS.
* **Database:** Firebase Firestore.

---

## 🚀 Getting Started (Local Development)

Because this project bridges Node.js and Python, you need to set up both environments to run the system locally.

### 1. Clone the Repository
```bash
git clone [https://github.com/shashank-95-95/Multi-camera-reid.git](https://github.com/shashank-95-95/Multi-camera-reid.git)
cd Multi-camera-reid