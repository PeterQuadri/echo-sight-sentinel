# EchoSight Sentinel - Deployment Checklist

This document outlines the steps to deploy the EchoSight Sentinel system in a production or real-world testing environment.

## 📡 1. Server Deployment (The Processing Unit)

The "Server" is the machine running the AI models (Audio Detection + Video Verification).

### ✅ Prerequisites
- [ ] **Hardware**: Machine with dedicated GPU (NVIDIA) recommended for faster inference, though CPU works (slower).
- [ ] **Network**: 
    - **Local Network**: Ensure the machine has a static local IP (e.g., `192.168.1.50`).
    - **Internet Access**: Use a service like **ngrok** to expose port 5000 if the client is not on the same Wi-Fi.
- [ ] **Environment Variables**:
    - Ensure `.env` is fully populated with:
        - `GROQ_API_KEY` (for Video verification)
        - `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_SANDBOX_NUMBER` (for WhatsApp alerts)
        - `USER_PHONE_NUMBER`

### 🚀 Launching the Server
 Run the server in "Production" mode (using the existing `websocket_server.py`):

```powershell
# 1. Activate Virtual Environment
.\venv\Scripts\activate

# 2. Run the Server
python websocket_server.py
```

---

## 💻 2. Client Deployment (The Eyes & Ears)

The "Client" is the monitoring device. This could be a Raspberry Pi, a laptop, or an IoT device with a Camera and Microphone.

### 🛠️ Client Software Requirements
You need to write or deploy a script on the client device that does the following:

1.  **Connects** to the Server via WebSocket (Socket.IO).
    - URL: `http://<SERVER_IP>:5000` (e.g., `http://192.168.1.15:5000` or `https://your-ngrok-url.ngrok-free.app`)

2.  **Captures Audio**:
    - Format: **Float32** (preferred) or Int16.
    - Sample Rate: **22050 Hz** (Must match server config in `realtime_detection_system.py`).
    - Chunk Size: Send data continuously (every ~0.5 seconds).
    - **Event Name**: `audio_data`

3.  **Captures Video**:
    - Format: **JPEG** images encoded as **Base64** strings.
    - Frame Rate: Send approx **1-5 FPS** (frames per second). No need for 30/60 FPS.
    - **Event Name**: `video_data`

### 📝 Example Client Implementation (Python)
You can use the `test_websocket_client.py` as a template, but you must modify it to capture **REAL** data instead of dummy noise.

**To use real hardware on the client:**
- Use `pyaudio` to capture microphone input.
- Use `cv2` (OpenCV) to capture webcam frames.

---

## 🌐 3. Network Configuration (Crucial)

If your Client and Server are on **different networks** (e.g., Server at home, Client at a different office), you **MUST** tunnel the connection.

**Option A: Ngrok (Easiest)**
1. On Server: `ngrok http 5000`
2. Copy the forwarding URL (e.g., `https://a1b2-c3d4.ngrok-free.app`).
3. On Client: Update the connection code to use this URL.

**Option B: Port Forwarding (Advanced)**
1. On your Router, forward port `5000` to the Server's local IP.
2. Client connects to your **Public IP**.

---

## 🌐 5. Firewall / Connection Issues? Use Ngrok (Recommended)

If the local connection fails (timeout), the easiest fix is to use **Ngrok** to create a secure tunnel.

### Step A: Start Ngrok (On Server)
1.  Open a new terminal.
2.  Run: `ngrok http 5000`
3.  Copy the **Forwarding URL** (e.g., `https://a1b2-c3d4.ngrok-free.app`).

### Step B: Connect Client
1.  Open `real_client_reference.py` on the client device.
2.  Update `SERVER_URL` with the Ngrok URL.
3.  Run the client. It should connect immediately, bypassing all firewalls!

1. Start Server.
2. Start Client (with real microphone/camera).
3. Make a noise near the Client (e.g., play a "glass breaking" sound effect from your phone).
4. **Server Terminal** should show: `🔊 AUDIO ALERT: GLASS_BREAKING`.
5. **WhatsApp** should receive a message with the AI's visual description of the scene.
