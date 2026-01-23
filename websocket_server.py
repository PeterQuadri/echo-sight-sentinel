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
from realtime_detection_system import RealTimeDetector, EmergencySoundCNN
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

# Global session manager: { sid: { 'detector': detector, 'analyzer': analyzer } }
user_sessions = {}

# Load system configuration once
load_dotenv()
GROQ_KEY = os.getenv("GROQ_API_KEY")
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'best_model.pth')
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
SANDBOX_NUM = os.getenv("TWILIO_SANDBOX_NUMBER")
YOUR_PHONE = os.getenv("USER_PHONE_NUMBER")
NOTIFIER = WhatsAppNotifier(TWILIO_SID, TWILIO_TOKEN, SANDBOX_NUM, YOUR_PHONE)

# Load and initialize the shared model globally to save memory
shared_model = None
try:
    print(f"📦 Pre-loading shared model from: {MODEL_PATH}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize architecture
    shared_model = EmergencySoundCNN(num_classes=4) # Assuming 4 based on CLASS_NAMES
    
    # Load state dict
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    shared_model.load_state_dict(checkpoint['model_state_dict'])
    shared_model.to(device)
    shared_model.eval()
    
    CLASS_NAMES = checkpoint.get('class_names', ['background', 'glass_breaking', 'gun_shots', 'screams'])
    print("✅ Shared model pre-loaded successfully!")
except Exception as e:
    print(f"❌ Critical Error pre-loading model: {e}")
    CLASS_NAMES = ['background', 'glass_breaking', 'gun_shots', 'screams']

DETECTOR_CONFIG = {
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

def get_or_create_session(sid):
    if sid not in user_sessions:
        print(f"🛠️ Creating isolated session for {sid}")
        analyzer = EmergencyVideoAnalyzer(groq_api_key=GROQ_KEY, headless=True)
        analyzer.start()
        
        detector = RealTimeDetector(
            model_path=MODEL_PATH,
            class_names=CLASS_NAMES,
            config=DETECTOR_CONFIG,
            video_analyzer=analyzer,
            whatsapp_notifier=NOTIFIER,
            socketio_server=sio,
            target_sid=sid,
            pretrained_model=shared_model
        )
        detector.start_headless()
        
        user_sessions[sid] = {
            'detector': detector,
            'analyzer': analyzer
        }
    return user_sessions[sid]

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
def disconnect(sid, reason=None):
    if sid in user_sessions:
        print(f"🧹 Cleaning up session for {sid} (Reason: {reason})")
        sess = user_sessions.pop(sid)
        sess['detector'].is_running = False
        sess['analyzer'].stop()
    print(f"❌ Client disconnected: {sid}")

@sio.event
def audio_data(sid, data):
    sess = get_or_create_session(sid)
    try:
        if isinstance(data, dict) and 'audio' in data:
             audio_bytes = base64.b64decode(data['audio'])
             audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
        elif isinstance(data, bytes):
            audio_array = np.frombuffer(data, dtype=np.float32)
        else:
            audio_array = np.frombuffer(data, dtype=np.float32)
        
        sess['detector'].audio_queue.put(audio_array)
    except Exception as e:
        print(f"Error processing audio data for {sid}: {e}")

@sio.event
def video_data(sid, data):
    sess = get_or_create_session(sid)
    try:
        if isinstance(data, dict) and 'frame' in data:
            frame_b64 = data['frame']
        else:
            frame_b64 = data
            
        sess['analyzer'].add_frame(frame_b64)
        
        # Emit BACK to the same SID for their own "Live Monitoring" view
        sio.emit('video_feed', frame_b64, to=sid)
    except Exception as e:
        print(f"Error processing video data for {sid}: {e}")

@sio.event
def trigger_manual_check(sid):
    sess = get_or_create_session(sid)
    print(f"⚡ Manual verification requested by {sid}")
    
    def run_check():
        res = sess['analyzer'].verify_alert(audio_class=None)
        sio.emit('status_update', res, to=sid)
        
    import threading
    threading.Thread(target=run_check, daemon=True).start()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print("\n" + "="*60)
    print(f"🚀 MULTI-SESSION SERVER RUNNING ON PORT {port}")
    print("="*60 + "\n")
    
    eventlet.wsgi.server(eventlet.listen(('0.0.0.0', port)), app)
