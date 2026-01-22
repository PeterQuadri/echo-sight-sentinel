# WebSocket Streaming & Testing Guide

This guide explains how to use the newly implemented WebSocket capabilities to stream audio and video to the EchoSight Sentinel system from a remote client.

## 1. Overview

We have separated the system into two potential components:
- **Server**: Receives audio/video feeds, runs the AI models (Audio Detection + Video Verification), and handles alerts.
- **Client**: Captures audio/video and sends it to the server.

## 2. New Files

- **`websocket_server.py`**: The new main entry point for the server. It listens on port 5000 for WebSocket connections (`/ws`).
- **`test_websocket_client.py`**: A test script that simulates a client by sending dummy audio (noise) and video (generated images) to the server.
- **`realtime_detection_system.py`**: Modified to support a "Headless" mode (no local microphone needed).

## 3. How to Test (Local Simulation)

You can verify the system works without deploying it to a separate machine yet.

### Step 1: Start the Server
Open a terminal and run the server. This will load the AI models (PyTorch + Groq) and wait for connections.

```powershell
python websocket_server.py
```

*Wait until you see "Rocket Server running on http://0.0.0.0:5000" and "✅ Detection system running in background".*

### Step 2: Run the Test Client
Open a **new** terminal window (keep the server running) and run the test client:

```powershell
python test_websocket_client.py
```

### Step 3: Observe Results

**In the Client Terminal:**
- You should see: `✅ Connected to EchoSight Sentinel Server`
- You will see continuous dots `.` or messages indicating audio/video frames are being sent.

**In the Server Terminal:**
- You will see: `✅ Client connected: ...`
- You should see the detection system processing audio chunks (e.g. `🎵 background: 99.0%`).
- Since we are sending random noise, it will likely detect "background".
- If the video analyzer is enabled, it might process frames periodically.

## 4. Deployment Protocol

When you are ready to deploy the actual client, your client application should implement the following SocketIO events:

**Connection:**
Connect to `http://<SERVER_IP>:5000`

**Events to Emit:**

1. **`audio_data`**:
   - Format: Binary bytes (Float32) OR JSON `{'audio': '<base64_encoded_bytes>'}`.
   - Sample Rate: 22050 Hz (recommended match to server config).
   - Sending frequency: continuously (chunks of ~1024 or 2048 samples).

2. **`video_data`**:
   - Format: Base64 string of a JPEG image OR JSON `{'frame': '<base64_string>'}`.
   - Sending frequency: ~5 FPS is sufficient for verification.

## 5. Troubleshooting

- **Port Conflicts**: If port 5000 is busy (e.g., used by the old WhatsApp server), stop the other process or change the port in `websocket_server.py`.
- **Model Paths**: Ensure `best_model.pth` is at `D:\DOCUMENTS\RAIN\AIML\Second_semester\Project\models\best_model.pth`.
- **Environment Variables**: Ensure `.env` is present for Twilio/Groq keys.
