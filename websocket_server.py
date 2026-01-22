import eventlet
eventlet.monkey_patch()
import socketio
import numpy as np
import base64
import os
import torch
import warnings
from dotenv import load_dotenv
from flask import Flask, render_template

# Import our system components
from realtime_detection_system import RealTimeDetector
from emergency_video_llm import EmergencyVideoAnalyzer
from whatsapp_notifier import WhatsAppNotifier
import whatsapp_server

# Suppress warnings
warnings.filterwarnings('ignore')

# Initialize Flask and SocketIO
app = Flask(__name__)
# Enable CORS for all domains (important for Render/Ngrok)
sio = socketio.Server(cors_allowed_origins='*', max_http_buffer_size=10*1024*1024) 
app.wsgi_app = socketio.WSGIApp(sio, app.wsgi_app)

# Global instances
detector = None
video_analyzer = None

# --- Routes ---
@app.route('/')
def index():
    """Serve the real-time dashboard"""
    return render_template('index.html')

@app.route('/health')
def health():
    return "OK", 200

# --- Socket Events ---

@sio.event
def connect(sid, environ):
    print(f"✅ Client connected: {sid}")

@sio.event
def disconnect(sid):
    print(f"❌ Client disconnected: {sid}")

@sio.event
def audio_data(sid, data):
    """
    Receive audio data chunk
    """
    global detector
    if detector and detector.is_running:
        try:
            # Decode data (bytes or json)
            if isinstance(data, dict) and 'audio' in data:
                 audio_bytes = base64.b64decode(data['audio'])
                 audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
            elif isinstance(data, bytes):
                audio_array = np.frombuffer(data, dtype=np.float32)
            else:
                audio_array = np.frombuffer(data, dtype=np.float32)
            
            # 1. Put in queue for processing
            detector.audio_queue.put(audio_array)

            # 2. BROADCAST Stats to Dashboard (Mocking live stats for visualization)
            # Ideally, the detector thread should emit this, but we can do a quick check here
            # Or better, we intercept the result in detector.predict
            
            # Since detector runs in a thread, we can't get result instantly here.
            # But we can allow the detector to access 'sio' to emit events.

        except Exception as e:
            print(f"Error processing audio data: {e}")

@sio.event
def video_data(sid, data):
    """
    Receive video frame
    """
    global video_analyzer
    if video_analyzer:
        try:
            # 1. Add to analyzer buffer
            if isinstance(data, dict) and 'frame' in data:
                frame_b64 = data['frame']
            else:
                frame_b64 = data
                
            video_analyzer.add_frame(frame_b64)
            
            # 2. BROADCAST to Dashboard (Relay the frame)
            # This allows the dashboard to see what the client sees
            sio.emit('video_feed', frame_b64)
            
        except Exception as e:
            print(f"Error processing video data: {e}")

@sio.event
def trigger_manual_check(sid):
    """Allow client to force a status check to verify video is working"""
    print(f"⚡ Manual verification requested by {sid}")
    if video_analyzer:
        # Run in thread to not block socket
        def run_check():
            print("👁️ Running manual video check...")
            res = video_analyzer.verify_alert(audio_class=None) # None = Status Check
            print(f"✅ Manual Check Result: {res}")
            sio.emit('status_update', res, to=sid)
            
        import threading
        threading.Thread(target=run_check, daemon=True).start()

def load_system():
    global detector, video_analyzer
    
    print("="*70)
    print("🌐 INITIALIZING ECHO SIGHT SENTINEL - WEBSOCKET SERVER")
    print("="*70)
    
    # Load env
    load_dotenv()
    
    # 1. Setup Video Analyzer (Headless)
    GROQ_KEY = os.getenv("GROQ_API_KEY")
    video_analyzer = EmergencyVideoAnalyzer(
        groq_api_key=GROQ_KEY, 
        headless=True  # IMPORTANT: No local camera
    )
    video_analyzer.start()
    
    # 2. Setup WhatsApp
    TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
    TWILIO_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
    SANDBOX_NUM = os.getenv("TWILIO_SANDBOX_NUMBER")
    YOUR_PHONE = os.getenv("USER_PHONE_NUMBER")
    notifier = WhatsAppNotifier(TWILIO_SID, TWILIO_TOKEN, SANDBOX_NUM, YOUR_PHONE)
    
    # 3. Setup Detector
    MODEL_PATH = r"D:\DOCUMENTS\RAIN\AIML\Second_semester\Project\models\best_model.pth"
    
    # Load class names (Simplified logic from main)
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        CLASS_NAMES = checkpoint.get('class_names', ['background', 'glass_breaking', 'gun_shots', 'screams'])
    except Exception as e:
        print(f"Warning loading model: {e}")
        CLASS_NAMES = ['background', 'glass_breaking', 'gun_shots', 'screams']

    config = {
        'sample_rate': 22050,
        'duration': 3,
        'chunk_size': 1024,
        'n_mels': 128,
        'confidence_threshold': 0.80,
        'temporal_window': 2,
        'energy_threshold': 0.002,
        'cooldown_seconds': 5,
        'log_dir': 'detection_logs'
    }
    
    detector = RealTimeDetector(
        model_path=MODEL_PATH,
        class_names=CLASS_NAMES,
        config=config,
        video_analyzer=video_analyzer,
        whatsapp_notifier=notifier,
        socketio_server=sio
    )
    
    detector.start_headless()

if __name__ == '__main__':
    load_system()
    
    # Get local IP
    import socket
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
    except:
        local_ip = "127.0.0.1"

    port = 5000
    print("\n" + "="*60)
    print(f"🚀 SERVER RUNNING! Client Connection Details:")
    print(f"   SAME DEVICE:   http://localhost:{port}")
    print(f"   LOCAL NETWORK: http://{local_ip}:{port}")
    print("="*60 + "\n")
    
    # Use eventlet for generic production-ready server
    eventlet.wsgi.server(eventlet.listen(('0.0.0.0', port)), app)
