# Dashboard Implementation & Render Deployment Plan

## Goal
Building a web-based dashboard to visualize real-time alerts, audio confidence levels, and video snapshots from the EchoSight Sentinel system. This dashboard will be hosted on the same Flask server and deployed to Render.

## 1. Technical Architecture

### Backend (`websocket_server.py`)
- **Serve Static Files**: modifying Flask to serve `index.html` at `/`.
- **Event Broadcasting**:
    - When `audio_data` is processed -> emit `audio_stats` (active class, confidence).
    - When `video_data` is received -> emit `video_feed` (base64 frame) to dashboard clients.
    - When `alert` is triggered -> emit `alert_event` to dashboard.

### Frontend (`templates/index.html`)
- **Connection**: Connects to the same WebSocket server.
- **Visuals**:
    - **Live Status**: "System Online" / "Monitoring".
    - **Audio Meter**: Visual bar showing confidence of current sound class.
    - **Latest Video**: Shows the most recent frame received from the client.
    - **Alert Log**: A list of recent confirmed emergencies.

## 2. Deployment on Render

Render is excellent for Python/Flask apps with WebSockets.

### Required Files
- **`Procfile`**: `web: gunicorn -k eventlet -w 1 websocket_server:app`
    - *Note*: `gunicorn` with `eventlet` worker is required for SocketIO production performance.
- **`requirements.txt`**: Ensure `gunicorn` and `eventlet` are listed.

## 3. Step-by-Step Implementation

1.  **Create Frontend**:
    - `mkdir templates`
    - Create `templates/index.html` (Modern dark UI).
    - `mkdir static` (for css/js if needed, or inline for simplicity).

2.  **Update Server**:
    - Modifying `websocket_server.py` to route `/` to `index.html`.
    - Add logic to relay `video_data` coming from *Client* -> *Server* -> *Dashboard*.

3.  **Prepare for Render**:
    - Create `Procfile`.
    - Create `render.yaml` (Infrastructure as Code).

4.  **Verification**:
    - Run server locally.
    - Open browser to `http://localhost:5000`.
    - Connect client.
    - Verify browser updates in real-time.

5.  **Final Production Config**:
    - **Get Render URL**: e.g., `https://echosightsentinel.onrender.com`.
    - **Update Client**: Replace `SERVER_URL` in `real_client_reference.py` with this Render URL.
    - **Run Client**: It will now connect securely to the cloud!
