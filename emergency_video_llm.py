import cv2
import base64
import json
from groq import Groq
from datetime import datetime
from typing import Dict, List
import time
import threading
from collections import deque
import numpy as np

class EmergencyVideoAnalyzer:
    """
    Real-time emergency video analysis using LLAMA vision models via Groq.
    Uses a rolling frame buffer to capture reactions "instantaneously".
    """
    
    def __init__(self, groq_api_key: str, video_source=0, model: str = "meta-llama/llama-4-scout-17b-16e-instruct", headless=False):
        """
        Args:
            groq_api_key: Groq API key
            video_source: Camera index or video path
            model: Groq vision model (11b is faster for real-time)
            headless: If True, does not initialize local camera (for web backend)
        """
        self.client = Groq(api_key=groq_api_key)
        self.model = model
        self.headless = headless
        
        # Frame Buffer (Last 2 seconds assuming 10 FPS)
        self.buffer_size = 20
        self.frame_buffer = deque(maxlen=self.buffer_size)
        
        # Video Capture
        self.cap = None
        if not self.headless:
            self.cap = cv2.VideoCapture(video_source)
        
        self.is_running = False
        self.lock = threading.Lock()
        
    def start(self):
        """Start the video capture thread"""
        if self.headless:
             print("📹 Video analysis initialized (Headless Mode)...")
             self.is_running = True
             return

        if not self.cap.isOpened():
            print("❌ Error: Could not open video source")
            return
            
        self.is_running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print("📹 Video analysis buffer started...")

    def stop(self):
        """Stop capture"""
        self.is_running = False
        if hasattr(self, 'thread'):
            self.thread.join()
        self.cap.release()

    def add_frame(self, frame_b64: str):
        """Manually add a frame from the browser (b64 string)"""
        try:
            img_bytes = base64.b64decode(frame_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is not None:
                with self.lock:
                    self.frame_buffer.append(frame)
        except Exception as e:
            print(f"Error adding frame to buffer: {e}")

    def _capture_loop(self):
        """Continuously update the frame buffer if NOT headless"""
        if self.headless or self.cap is None:
            return
            
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame_buffer.append(frame)
            time.sleep(0.1) # 10 FPS is enough for reaction capture

    def encode_frame(self, frame) -> str:
        """Base64 encode for API"""
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.b64encode(buffer).decode('utf-8')

    def verify_alert(self, audio_class: str = None) -> Dict:
        """
        Analyzes a burst of frames. 
        If audio_class is provided, it confirms an alert.
        If audio_class is None, it provides a general status description.
        """
        if audio_class:
            print(f"🔍 Visually verifying audio alert: {audio_class.upper()}...")
        else:
            print(f"👁️ Performing remote status check...")
        
        # Grab frames from buffer (past 2 seconds) and 5 more now
        burst_frames = []
        with self.lock:
            # Take every 4th frame from buffer for diversity (covers ~2 secs)
            buffer_list = list(self.frame_buffer)
            for i in range(0, len(buffer_list), 4):
                burst_frames.append(buffer_list[i])
        
        # Capture 5 more "present" frames (if camera available)
        if self.cap is not None:
            for _ in range(5):
                ret, frame = self.cap.read()
                if ret:
                    burst_frames.append(frame)
                time.sleep(0.2)
        else:
            # Headless mode: just use what is in the buffer (most recent)
            pass

        if not burst_frames:
            return {"decision": "Error", "reasoning": "No frames captured"}

        # Analyze a subset (e.g., 5 frames) to keep it fast but accurate
        step = max(1, len(burst_frames) // 5)
        analysis_frames = burst_frames[::step][:5]
        
        try:
            # Prompt tailored to confirm the SPECIFIC class or just describe
            if audio_class:
                prompt = f"""You are a security AI. An audio system detected: {audio_class.upper()}.
Check these image sequence for VISUAL confirmation of this emergency. 

If there IS an emergency, identify the specific threat (weapons, fighting, panic).
If there IS NOT an emergency, describe the scene calmly (e.g., "a person typing on a laptop", "a boy sitting still", "an empty room").

Respond ONLY with JSON:
{{ 
  "decision": "True Emergency" or "False Positive",
  "confidence": 0.0-1.0,
  "reasoning": "A one-sentence description of exactly what you see happening in the video."
}}"""
            else:
                prompt = """You are a home security companion. 
The user is checking in remotely. Describe exactly what is happening in this image sequence.
Be specific about who is there (if anyone) and what they are doing. 

Respond ONLY with JSON:
{{ 
  "decision": "Status Update",
  "confidence": 1.0,
  "reasoning": "A descriptive summary of the current scene (e.g., 'The living room is empty', 'Two children are watching TV')."
}}"""

            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            
            # Add images to the message
            for frame in analysis_frames:
                b64 = self.encode_frame(frame)
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                })

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=200
            )

            response_text = response.choices[0].message.content
            
            # Robust JSON extraction
            import re
            try:
                # Find all JSON-like objects {...}
                json_blocks = re.findall(r'\{.*?\}', response_text, re.DOTALL)
                
                if json_blocks:
                    # Parse each block
                    parsed_objects = []
                    for block in json_blocks:
                        try:
                            parsed_objects.append(json.loads(block))
                        except:
                            continue
                    
                    if not parsed_objects:
                        raise ValueError("No valid JSON objects found")
                        
                    # Consolidate
                    if len(parsed_objects) > 1:
                        # Take the first one as base, but combine all reasonings
                        result = parsed_objects[0]
                        all_reasons = []
                        for obj in parsed_objects:
                            r = obj.get('reasoning', '')
                            if r and r not in all_reasons:
                                all_reasons.append(r)
                        result['reasoning'] = " ".join(all_reasons)
                    else:
                        result = parsed_objects[0]
                else:
                    # Fallback to direct list check if provided by model
                    list_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                    if list_match:
                        result = json.loads(list_match.group(0))
                        if isinstance(result, list) and len(result) > 0:
                            all_reasons = [i.get('reasoning', '') for i in result if isinstance(i, dict) and i.get('reasoning')]
                            result = result[0]
                            result['reasoning'] = " ".join(list(set(all_reasons)))
                    else:
                        raise ValueError("No JSON found in response")
                    
            except (json.JSONDecodeError, ValueError) as e:
                print(f"❌ JSON parse failed. Raw response: {response_text}")
                raise e

            print(f"📹 Video Analysis: {result['decision']} ({result.get('confidence', 0)*100:.0f}%)")
            return result

        except Exception as e:
            print(f"❌ Video analysis failed: {e}")
            return {"decision": "Error", "reasoning": str(e)}
