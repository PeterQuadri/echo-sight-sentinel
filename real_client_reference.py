import socketio
import time
import numpy as np
import base64
import cv2
import threading
import pyaudio

# ---------------- CONFIGURATION ----------------
# OPTION A: Local Network (If Firewall is configured)
# SERVER_URL = 'http://192.168.1.223:5000' 

# OPTION B: Ngrok (Recommended for reliability)
# Run 'ngrok http 5000' on server and paste URL here:
SERVER_URL = 'https://multilateral-lilla-proequality.ngrok-free.dev' 

AUDIO_RATE = 22050
AUDIO_CHUNK = 1024
VIDEO_FPS = 5
# -----------------------------------------------

sio = socketio.Client()
is_running = True

@sio.event
def connect():
    print("✅ Connected to EchoSight Sentinel Server")

@sio.event
def disconnect():
    print("❌ Disconnected from server")

@sio.event
def status_update(data):
    print(f"\n📩 Message from Server: {data}")

def audio_stream_task():
    print("🎤 Initializing Microphone...")
    p = pyaudio.PyAudio()
    
    try:
        stream = p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=AUDIO_RATE,
                        input=True,
                        frames_per_buffer=AUDIO_CHUNK)
        
        print("🎤 Audio Stream Started!")
        
        while is_running and sio.connected:
            try:
                data = stream.read(AUDIO_CHUNK, exception_on_overflow=False)
                # Send raw bytes (float32)
                sio.emit('audio_data', data)
            except Exception as e:
                print(f"Audio Error: {e}")
                break
                
    except Exception as e:
        print(f"❌ Failed to open microphone: {e}")
    finally:
        if 'stream' in locals():
            stream.stop_stream()
            stream.close()
        p.terminate()

def video_stream_task():
    print("📷 Initializing Camera...")
    cap = cv2.VideoCapture(0) # 0 = Default Camera
    
    if not cap.isOpened():
        print("❌ Failed to open camera!")
        return

    print("📷 Video Stream Started!")
    
    while is_running and sio.connected:
        try:
            ret, frame = cap.read()
            if ret:
                # Resize to reduce bandwidth (e.g., 640px width)
                # frame = cv2.resize(frame, (640, 480))
                
                # Compress to JPEG
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                b64_frame = base64.b64encode(buffer).decode('utf-8')
                
                sio.emit('video_data', b64_frame)
                
                # Limit FPS
                time.sleep(1.0 / VIDEO_FPS)
            else:
                print("Warning: Empty frame")
                time.sleep(1)
        except Exception as e:
            print(f"Video Error: {e}")
            break
            
    cap.release()

def main():
    global is_running
    try:
        print(f"Attempting connection to {SERVER_URL}...")
        sio.connect(SERVER_URL)
        
        # Start Threads
        t1 = threading.Thread(target=audio_stream_task, daemon=True)
        t2 = threading.Thread(target=video_stream_task, daemon=True)
        
        t1.start()
        t2.start()
        
        # Keep main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping client...")
        is_running = False
        sio.disconnect()
    except Exception as e:
        print(f"\n❌ Client Error: {e}")

if __name__ == "__main__":
    main()
