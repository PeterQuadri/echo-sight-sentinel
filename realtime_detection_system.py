"""
Real-Time Emergency Sound Detection System
Continuously monitors audio and alerts on detected emergencies
"""

import torch
import torch.nn.functional as F
import numpy as np
import librosa
import queue
import threading
from datetime import datetime
from pathlib import Path
import json
import time
from collections import deque
import warnings
from emergency_video_llm import EmergencyVideoAnalyzer
from whatsapp_notifier import WhatsAppNotifier
import whatsapp_server
from dotenv import load_dotenv
import os
warnings.filterwarnings('ignore')

# Import your model architecture
# Make sure this matches your training script
class EmergencySoundCNN(torch.nn.Module):
    """CNN model - must match training architecture exactly"""
    def __init__(self, num_classes=4, dropout_rate=0.5):
        super(EmergencySoundCNN, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        self.dropout1 = torch.nn.Dropout2d(0.1)
        
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.relu2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(2, 2)
        self.dropout2 = torch.nn.Dropout2d(0.2)
        
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.relu3 = torch.nn.ReLU()
        self.pool3 = torch.nn.MaxPool2d(2, 2)
        self.dropout3 = torch.nn.Dropout2d(0.2)
        
        self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(256)
        self.relu4 = torch.nn.ReLU()
        self.pool4 = torch.nn.MaxPool2d(2, 2)
        self.dropout4 = torch.nn.Dropout2d(0.3)
        
        self.global_avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        
        self.fc1 = torch.nn.Linear(256, 128)
        self.bn_fc1 = torch.nn.BatchNorm1d(128)
        self.relu_fc1 = torch.nn.ReLU()
        self.dropout_fc1 = torch.nn.Dropout(dropout_rate)
        
        self.fc2 = torch.nn.Linear(128, 64)
        self.bn_fc2 = torch.nn.BatchNorm1d(64)
        self.relu_fc2 = torch.nn.ReLU()
        self.dropout_fc2 = torch.nn.Dropout(dropout_rate * 0.5)
        
        self.fc3 = torch.nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.dropout1(self.pool1(self.relu1(self.bn1(self.conv1(x)))))
        x = self.dropout2(self.pool2(self.relu2(self.bn2(self.conv2(x)))))
        x = self.dropout3(self.pool3(self.relu3(self.bn3(self.conv3(x)))))
        x = self.dropout4(self.pool4(self.relu4(self.bn4(self.conv4(x)))))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout_fc1(self.relu_fc1(self.bn_fc1(self.fc1(x))))
        x = self.dropout_fc2(self.relu_fc2(self.bn_fc2(self.fc2(x))))
        x = self.fc3(x)
        return x


class RealTimeDetector:
    """
    Real-time emergency sound detection system
    """
    
    def __init__(self, model_path, class_names, config, video_analyzer=None, whatsapp_notifier=None, socketio_server=None, target_sid=None, pretrained_model=None):
        """
        Initialize the detector
        
        Args:
            model_path: Path to trained model (.pth file)
            class_names: List of class names
            config: Configuration dictionary
            socketio_server: SocketIO server instance for broadcasting events
            target_sid: If provided, emits will be targeted to this specific socket ID
            pretrained_model: Optional pre-loaded model object to avoid disk I/O and memory duplication
        """
        self.class_names = class_names
        self.config = config
        self.video_analyzer = video_analyzer
        self.whatsapp_notifier = whatsapp_notifier
        self.sio = socketio_server
        self.sid = target_sid
        
        # Audio parameters
        self.sr = config.get('sample_rate', 22050)
        self.duration = config.get('duration', 1)
        self.chunk_size = config.get('chunk_size', 1024)
        self.n_mels = config.get('n_mels', 128)
        self.n_samples = int(self.sr * self.duration)
        
        # Detection parameters
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        self.temporal_window = config.get('temporal_window', 1)
        self.cooldown_seconds = config.get('cooldown_seconds', 5)
        self.energy_threshold = config.get('energy_threshold', 0.005)
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model or use pretrained instance
        if pretrained_model is not None:
            self.model = pretrained_model
            print("🚀 Using shared pretrained model instance")
        else:
            print(f"📦 Loading model from: {model_path}")
            self.model = self.load_model(model_path)
            
        self.model.eval()
        self.model.to(self.device)
        print("✅ Model ready!")
        
        # Detection history for temporal filtering
        self.detection_history = deque(maxlen=self.temporal_window)
        
        # Audio queue for threading
        self.audio_queue = queue.Queue()
        
        # State management
        self.is_running = False
        self.last_alert_time = {}
        
        # Logging
        self.log_dir = Path(config.get('log_dir', 'logs'))
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / f"detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'detections': {name: 0 for name in class_names},
            'start_time': None
        }
    
    def load_model(self, model_path):
        """Load trained model"""
        model = EmergencySoundCNN(num_classes=len(self.class_names))
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        return model
    
    def preprocess_audio(self, audio):
        """
        Preprocess audio chunk
        Same preprocessing as training
        """
        # Ensure correct length
        if len(audio) < self.n_samples:
            audio = np.pad(audio, (0, self.n_samples - len(audio)))
        else:
            audio = audio[:self.n_samples]
        
        # Normalize - Moved this to AFTER energy check in predict()
        # if np.max(np.abs(audio)) > 0:
        #     audio = audio / np.max(np.abs(audio))
        
        return audio
    
    def extract_features(self, audio):
        """Extract mel-spectrogram features"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_mels=self.n_mels,
            fmax=8000,
            hop_length=512
        )
        
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Improved normalization: only normalize if there's significant content
        db_range = mel_spec_db.max() - mel_spec_db.min()
        if db_range > 10:
            mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (db_range + 1e-8)
        else:
            # If very quiet, just scale relative to a "safe" floor
            mel_spec_db = (mel_spec_db + 80) / 80
            mel_spec_db = np.clip(mel_spec_db, 0, 1)
        
        # Convert to tensor
        features = torch.FloatTensor(mel_spec_db).unsqueeze(0).unsqueeze(0)
        return features
    
    def predict(self, audio):
        """
        Make prediction on audio chunk
        
        Returns:
            class_idx: Predicted class index
            class_name: Predicted class name
            confidence: Confidence score
            all_probs: All class probabilities
        """
        # Preprocess (Resamples/Pads/Trims but doesn't normalize yet)
        audio = self.preprocess_audio(audio)
        
        # Energy threshold check on RAW audio (RMS)
        rms = np.sqrt(np.mean(audio**2))
        if rms < self.energy_threshold:
            num_classes = len(self.class_names)
            bg_probs = np.zeros(num_classes)
            bg_probs[0] = 1.0 # Background is usually the first class
            return 0, self.class_names[0], 1.0, bg_probs
        
        # Normalize now for the model
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # Extract features
        features = self.extract_features(audio).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(features)
            probabilities = F.softmax(outputs, dim=1)
        
        # Get prediction
        confidence, predicted_idx = probabilities.max(1)
        confidence = confidence.item()
        predicted_idx = predicted_idx.item()
        class_name = self.class_names[predicted_idx]
        all_probs = probabilities.cpu().numpy()[0]
        
        return predicted_idx, class_name, confidence, all_probs
    
    def temporal_filter(self, class_idx, confidence):
        """
        Apply temporal filtering to reduce false positives
        
        Returns:
            is_emergency: True if emergency confirmed
            filtered_class: Class that passed filter
            avg_confidence: Average confidence over window
        """
        # Add to history
        self.detection_history.append((class_idx, confidence))
        
        # Check if we have enough history
        if len(self.detection_history) < self.temporal_window:
            return False, None, 0.0
        
        # Get recent detections
        recent_classes = [det[0] for det in self.detection_history]
        recent_confidences = [det[1] for det in self.detection_history]
        
        # Find most common class in the window
        from collections import Counter
        class_counts = Counter(recent_classes)
        most_common_class, count = class_counts.most_common(1)[0]
        
        # Stricter check:
        # 1. Majority agrees
        # 2. It's not background (0)
        # 3. The CURRENT detection (class_idx) is the same as the most common one
        #    This prevents a previous emergency from triggering an alert when we are now hearing background.
        if count >= self.temporal_window and most_common_class != 0 and most_common_class == class_idx:
            avg_confidence = np.mean([conf for cls, conf in self.detection_history if cls == most_common_class])
            
            if avg_confidence >= self.confidence_threshold:
                return True, most_common_class, avg_confidence
        
        return False, None, 0.0
    
    def in_cooldown(self, class_name):
        """Check if class is in cooldown period"""
        if class_name not in self.last_alert_time:
            return False
        
        time_since_last = time.time() - self.last_alert_time[class_name]
        return time_since_last < self.cooldown_seconds
    
    def trigger_alert(self, class_name, confidence):
        """
        Trigger emergency alert with Video Verification
        """
        timestamp = datetime.now()
        
        # 1. Log the initial audio detection
        print("\n" + "!"*70)
        print(f"🔊 AUDIO ALERT: {class_name.upper()} ({confidence*100:.1f}%)")
        print("!"*70)

        # 2. Trigger Video Verification in background
        if self.video_analyzer:
            def verify():
                res = self.video_analyzer.verify_alert(class_name)
                if res.get('decision') == 'True Emergency':
                    print("\n" + "="*70)
                    print("🚨 EMERGENCY CONFIRMED BY VIDEO! 🚨")
                    print("="*70)
                    print(f"⏰ Time:      {timestamp.strftime('%H:%M:%S')}")
                    print(f"📢 Event:     {class_name.upper()}")
                    print(f"👀 Reasoning: {res.get('reasoning')}")
                    print("="*70 + "\n")
                    
                    # BROADCAST ALERT TO DASHBOARD
                    if self.sio:
                        self.sio.emit('alert_event', res, to=self.sid)

                    # 3. Send WhatsApp Alert
                    if self.whatsapp_notifier:
                        self.whatsapp_notifier.send_alert(
                            class_name, 
                            confidence, 
                            res.get('reasoning'), 
                            timestamp.strftime('%Y-%m-%d %H:%M:%S')
                        )
                else:
                    print(f"✅ Video: False Positive")
                    print(f"👀 AI Observation: {res.get('reasoning')}")
                    
                    # Still log to dashboard as info
                    if self.sio:
                        self.sio.emit('status_update', res, to=self.sid)
            
            threading.Thread(target=verify, daemon=True).start()
        else:
            # Fallback if no video analyzer
            print(f"🚨 ALERT (No Video): {class_name.upper()}")

        # Log to file
        log_entry = f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} | {class_name} | {confidence:.4f}\n"
        with open(self.log_file, 'a') as f:
            f.write(log_entry)
        
        # Update statistics
        self.stats['detections'][class_name] += 1
        
        # Update cooldown
        self.last_alert_time[class_name] = time.time()
        
        # Here you can add:
        # - Send SMS/Email notification
        # - Play alarm sound
        # - Send to monitoring system
        # - Save audio clip
        # - Trigger other actions
    
    
    def process_audio_stream(self):
        """
        Main processing loop
        Continuously processes audio from queue
        """
        audio_buffer = np.array([])
        
        print("👂 Listening for emergency sounds...")
        print(f"   Confidence threshold: {self.confidence_threshold*100}%")
        print(f"   Temporal window: {self.temporal_window} detections")
        print(f"   Cooldown period: {self.cooldown_seconds} seconds")
        print("\nPress Ctrl+C to stop\n")
        
        while self.is_running:
            try:
                # Get audio chunk
                chunk = self.audio_queue.get(timeout=1)
                audio_buffer = np.concatenate([audio_buffer, chunk])
                
                # Process when we have enough audio
                if len(audio_buffer) >= self.n_samples:
                    # Make prediction
                    class_idx, class_name, confidence, all_probs = self.predict(audio_buffer)
                    
                    self.stats['total_processed'] += 1
                    
                    # Display current detection
                    probs_str = " | ".join([f"{name}: {prob*100:.1f}%" for name, prob in zip(self.class_names, all_probs)])
                    print(f"🎵 {probs_str}", end='\r')
                    
                    # BROADCAST TO DASHBOARD
                    if self.sio:
                        # Convert numpy probabilities to standard list of floats
                        probs_list = [float(p) for p in all_probs]
                        self.sio.emit('audio_stats', {
                            'class_name': class_name,
                            'confidence': float(confidence),
                            'all_probs': probs_list
                        }, to=self.sid)
                    
                    # Apply temporal filtering
                    is_emergency, filtered_class, avg_confidence = self.temporal_filter(class_idx, confidence)
                    
                    # Trigger alert if emergency confirmed
                    if is_emergency:
                        filtered_class_name = self.class_names[filtered_class]
                        
                        # Check cooldown
                        if not self.in_cooldown(filtered_class_name):
                            self.trigger_alert(filtered_class_name, avg_confidence)
                            
                            # Clear detection history AND audio buffer to avoid re-triggering on same sound
                            self.detection_history.clear()
                            audio_buffer = np.array([]) 
                            
                            # DON'T sleep here! Cooldown handles alert frequency, 
                            # we must keep processing the queue or it will back up.
                    
                    # Proper Sliding Window: keep the last (n_samples - stride)
                    # We stride by 1 second (self.sr) to update predictions every second
                    stride_samples = int(self.sr)
                    if len(audio_buffer) >= self.n_samples:
                         audio_buffer = audio_buffer[stride_samples:]
                    else:
                         audio_buffer = np.array([])
                
                # Prevent queue overflow (lag protection)
                if self.audio_queue.qsize() > 20: 
                    # If we're behind by more than ~1 second, drop oldest data
                    while self.audio_queue.qsize() > 5:
                        try:
                            self.audio_queue.get_nowait()
                        except queue.Empty:
                            break
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"\n❌ Error processing audio: {e}")
                continue
    
    def start_headless(self):
        """Start detection without local audio capture (for server mode)"""
        print("="*70)
        print("🚀 STARTING EMERGENCY SOUND DETECTION SYSTEM (HEADLESS/SERVER MODE)")
        print("="*70)
        print(f"\n⚙️  Configuration:")
        print(f"   Sample Rate: {self.sr} Hz")
        print(f"   Duration: {self.duration} seconds")
        print(f"   Classes: {', '.join(self.class_names)}")
        print(f"   Device: {self.device}")
        
        self.is_running = True
        self.stats['start_time'] = time.time()
        
        # Start processing thread
        processing_thread = threading.Thread(target=self.process_audio_stream)
        processing_thread.daemon = True
        processing_thread.start()
        
        print("✅ Detection system running in background (waiting for socket input)...")

    def print_statistics(self):
        """Print detection statistics"""
        if self.stats['start_time'] is None:
            return 
            
        runtime = time.time() - self.stats['start_time']
        
        print("\n" + "="*70)
        print("📊 DETECTION STATISTICS")
        print("="*70)
        print(f"Runtime: {runtime/60:.2f} minutes")
        print(f"Audio chunks processed: {self.stats['total_processed']}")
        print(f"\nDetections:")
        for class_name, count in self.stats['detections'].items():
            if class_name != 'background':
                print(f"  {class_name}: {count}")
        print(f"\nLog file: {self.log_file}")
        print("="*70)


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Main function"""
    print("="*70)
    print("🎯 EMERGENCY SOUND DETECTION - REAL-TIME SYSTEM")
    print("="*70)
    
    # ============================================================
    # CONFIGURATION
    # ============================================================
    
    MODEL_PATH = r"D:\DOCUMENTS\RAIN\AIML\Second_semester\Project\models\best_model.pth"
    
    # Load class names from model checkpoint
    print("\n📦 Loading model to detect class names...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    if 'class_names' in checkpoint:
        CLASS_NAMES = checkpoint['class_names']
        print(f"✅ Found {len(CLASS_NAMES)} classes: {CLASS_NAMES}")
    else:
        # Detect from model shape
        num_classes = checkpoint['model_state_dict']['fc3.bias'].shape[0]
        print(f"⚠️  Class names not in checkpoint. Detected {num_classes} classes.")
        
        if num_classes == 4:
            CLASS_NAMES = ['background', 'glass_breaking', 'gun_shots', 'screams']
        elif num_classes == 5:
            CLASS_NAMES = ['background', 'glass_breaking', 'gun_shots', 'screams', 'forced_entry']
        else:
            CLASS_NAMES = [f'class_{i}' for i in range(num_classes)]
        
        print(f"   Using: {CLASS_NAMES}")
    
    config = {
        'sample_rate': 22050,
        'duration': 3,                 # Increased to match training data
        'chunk_size': 1024,
        'n_mels': 128,
        'confidence_threshold': 0.80,  # Set to 80% as requested
        'temporal_window': 2,          # Still require a bit of consistency
        'energy_threshold': 0.002,     # Lowered to pick up softer live sounds
        'cooldown_seconds': 5,
        'log_dir': 'detection_logs'
    }
    
    # ============================================================
    # INITIALIZE AND START DETECTOR
    # ============================================================
    
    try:
        # Load environment variables from .env file
        load_dotenv()

        # Initialize Video Analyzer (using environment variables)
        GROQ_KEY = os.getenv("GROQ_API_KEY")
        if not GROQ_KEY:
            print("⚠️ WARNING: GROQ_API_KEY not found in .env file!")
        else:
            print(f"✅ GROQ_API_KEY loaded ({GROQ_KEY[:5]}...{GROQ_KEY[-5:]})")
            
        video_analyzer = EmergencyVideoAnalyzer(groq_api_key=GROQ_KEY)
        video_analyzer.start()

        # Initialize WhatsApp Notifier (using environment variables)
        TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
        TWILIO_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
        SANDBOX_NUM = os.getenv("TWILIO_SANDBOX_NUMBER")
        YOUR_PHONE = os.getenv("USER_PHONE_NUMBER")
        
        if not all([TWILIO_SID, TWILIO_TOKEN, SANDBOX_NUM, YOUR_PHONE]):
            print("⚠️ WARNING: Twilio credentials missing in .env file!")
            print(f"   SID: {'OK' if TWILIO_SID else 'MISSING'}")
            print(f"   Token: {'OK' if TWILIO_TOKEN else 'MISSING'}")
            print(f"   Sandbox: {'OK' if SANDBOX_NUM else 'MISSING'}")
            print(f"   Phone: {'OK' if YOUR_PHONE else 'MISSING'}")
        else:
            print(f"✅ Twilio Credentials loaded for {YOUR_PHONE}")
            
        notifier = WhatsAppNotifier(TWILIO_SID, TWILIO_TOKEN, SANDBOX_NUM, YOUR_PHONE)

        # START WHATSAPP WEBHOOK SERVER (for remote queries)
        whatsapp_server.start_server(video_analyzer, notifier, port=5000)

        detector = RealTimeDetector(
            model_path=MODEL_PATH,
            class_names=CLASS_NAMES,
            config=config,
            video_analyzer=video_analyzer,
            whatsapp_notifier=notifier
        )
        
        try:
            detector.start()
        finally:
            video_analyzer.stop()
        
    except FileNotFoundError:
        print(f"\n❌ Model file not found: {MODEL_PATH}")
        print("   Please train the model first!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
