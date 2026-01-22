import socketio
import time
import numpy as np
import base64
import cv2  # Just for creating a dummy image, or we can hardcode a b64 string
import threading

# Initialize SocketIO Client
sio = socketio.Client()

SERVER_URL = 'http://localhost:5000'

@sio.event
def connect():
    print("✅ Connected to EchoSight Sentinel Server")

@sio.event
def disconnect():
    print("❌ Disconnected from server")

@sio.event
def connect_error(data):
    print(f"❌ Connection failed: {data}")

@sio.event
def status_update(data):
    print("\n" + "="*50)
    print("📨 RECEIVED STATUS UPDATE FROM SERVER")
    print("="*50)
    print(f"Decision: {data.get('decision')}")
    print(f"Reasoning: {data.get('reasoning')}")
    print("="*50 + "\n")

def generate_dummy_audio(duration=1.0, sr=22050):
    """Generate 1 second of white noise"""
    audio = np.random.uniform(-0.1, 0.1, int(sr * duration)).astype(np.float32)
    return audio.tobytes()

def generate_dummy_frame_b64():
    """Generate a dummy JPEG frame (random noise)"""
    # Create random images (100x100)
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def stream_audio():
    print("🎵 Starting audio stream simulation...")
    while True:
        try:
            # Send 1 second of audio
            audio_bytes = generate_dummy_audio(duration=1.0)
            
            # Use emit with binary data
            sio.emit('audio_data', audio_bytes)
            print(".", end='', flush=True)
            
            time.sleep(1.0) # Wait for 1 second real-time
        except Exception as e:
            print(f"Audio stream error: {e}")
            break

def stream_video():
    print("📹 Starting video stream simulation...")
    while True:
        try:
            # Send frames at a low FPS (e.g. 5fps)
            frame_b64 = generate_dummy_frame_b64()
            sio.emit('video_data', frame_b64)
            
            time.sleep(0.2) # 5 FPS
        except Exception as e:
            print(f"Video stream error: {e}")
            break

def main():
    try:
        print(f"Attempting connection to {SERVER_URL}...")
        sio.connect(SERVER_URL)
        
        # Start Threads
        audio_thread = threading.Thread(target=stream_audio)
        video_thread = threading.Thread(target=stream_video)
        
        audio_thread.daemon = True
        video_thread.daemon = True
        
        audio_thread.start()
        video_thread.start()
        
        # Keep alive and trigger check
        time.sleep(5)
        print("\n⚡ Triggering manual VIDEO check in 3 seconds...")
        time.sleep(3)
        sio.emit('trigger_manual_check')
        print("⚡ Signal sent! Waiting for analysis result...")
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping test...")
        sio.disconnect()
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        # If connection fails, ensure we exit
        import sys
        sys.exit(1)

if __name__ == "__main__":
    main()
