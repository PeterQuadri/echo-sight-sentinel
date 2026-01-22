from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
import threading

app = Flask(__name__)

# Global references set by the main detector
video_analyzer = None
whatsapp_notifier = None

@app.route("/whatsapp", methods=['POST'])
def whatsapp_reply():
    """Handle incoming WhatsApp messages"""
    incoming_msg = request.values.get('Body', '').lower().strip()
    sender = request.values.get('From', '')
    
    print(f"📥 Received WhatsApp from {sender}: {incoming_msg}")
    
    resp = MessagingResponse()
    
    # List of keywords that trigger a status check
    status_keywords = ["status", "check", "how", "situation", "house", "monitoring", "update", "happening"]
    
    is_status_query = any(kw in incoming_msg for kw in status_keywords)
    
    if is_status_query:
        if video_analyzer:
            print("👁️ TRIGGER: Remote Status Check requested.")
            def perform_check():
                try:
                    result = video_analyzer.verify_alert(audio_class=None)
                    reasoning_text = result.get('reasoning', "I can see the room, but I couldn't generate a detailed description right now.")
                    report = f"🏠 *Home Status Update*\n\n{reasoning_text}"
                    if whatsapp_notifier:
                        print("📡 SENDING: Remote Status Report...")
                        whatsapp_notifier.send_message(report)
                    else:
                        print("❌ ERROR: WhatsApp notifier is None!")
                except Exception as e:
                    print(f"❌ ERROR: Status check thread failed: {e}")
            
            threading.Thread(target=perform_check, daemon=True).start()
            return str(resp.message("👁️ One moment, I'm checking the live feed for you..."))
        else:
            print("❌ ERROR: Video analyzer not initialized in server!")
            return str(resp.message("❌ System Error: Video analyzer is offline."))
            
    # Default help message
    print(f"❓ Unknown command received. (Keywords looked for: {status_keywords})")
    reply = "🤖 *EchoSight Sentinel Bot*\n\nI didn't quite catch that. Send 'status', 'check', or 'how is the house' to get a real-time visual update!"
    return str(resp.message(reply))

def start_server(analyzer, notifier, port=5000):
    """Run the Flask server in a thread"""
    global video_analyzer, whatsapp_notifier
    video_analyzer = analyzer
    whatsapp_notifier = notifier
    
    # Run Flask without the reloader to avoid issues in threads
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False), daemon=True).start()
    print(f"🌐 WhatsApp Webhook Server started on port {port}")

if __name__ == "__main__":
    # For standalone testing
    app.run(port=5000, debug=True)
