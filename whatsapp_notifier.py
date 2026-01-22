from twilio.rest import Client
import logging

class WhatsAppNotifier:
    """
    Handles sending WhatsApp emergency alerts via Twilio Sandbox.
    """
    
    def __init__(self, account_sid, auth_token, sandbox_number, target_number):
        """
        Args:
            account_sid: Twilio Account SID
            auth_token: Twilio Auth Token
            sandbox_number: The Twilio sandbox number (e.g. +14155238886)
            target_number: Your personal phone number registered with the sandbox
        """
        self.client = Client(account_sid, auth_token)
        self.from_number = f"whatsapp:{sandbox_number}"
        self.to_number = f"whatsapp:{target_number}"
        
    def send_alert(self, event_type, confidence, reasoning, timestamp):
        """
        Send a formatted emergency alert.
        """
        message_body = (
            f"🚨 *EMERGENCY ALERT CONFIRMED* 🚨\n\n"
            f"*Event:* {event_type.upper()}\n"
            f"*Time:* {timestamp}\n"
            f"*Audio Confidence:* {confidence*100:.1f}%\n"
            f"*Visual Observation:* {reasoning}\n\n"
            f"⚠️ Please check your live feed immediately!"
        )
        
        try:
            print(f"📡 Sending message From: {self.from_number} To: {self.to_number}")
            message = self.client.messages.create(
                body=message_body,
                from_=self.from_number,
                to=self.to_number
            )
            print(f"✅ WhatsApp alert sent! SID: {message.sid}")
            return True
        except Exception as e:
            print(f"❌ Failed to send WhatsApp alert: {e}")
            return False

    def send_message(self, body):
        """
        Send a non-formatted plain text message.
        """
        try:
            print(f"📡 Sending Reply To: {self.to_number}")
            message = self.client.messages.create(
                body=body,
                from_=self.from_number,
                to=self.to_number
            )
            return True
        except Exception as e:
            print(f"❌ Failed to send WhatsApp reply: {e}")
            return False

if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    load_dotenv()
    
    # Test script using environment variables
    ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
    AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
    SANDBOX_NUM = os.getenv("TWILIO_SANDBOX_NUMBER")
    TARGET_NUM = os.getenv("USER_PHONE_NUMBER")
    
    if not all([ACCOUNT_SID, AUTH_TOKEN, SANDBOX_NUM, TARGET_NUM]):
        print("❌ Error: Missing Twilio credentials in .env file.")
    else:
        notifier = WhatsAppNotifier(ACCOUNT_SID, AUTH_TOKEN, SANDBOX_NUM, TARGET_NUM)
        print(f"Using Twilio From: {notifier.from_number} To: {notifier.to_number}")
        notifier.send_alert("Test Alert", 0.99, "This is a manual integration test using .env.", "2026-01-05 19:45:00")
