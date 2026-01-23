import os
import base64
from supabase import create_client, Client
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

url: str = os.getenv("SUPABASE_URL")
key: str = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)

def upload_evidence(image_b64: str) -> str:
    """
    Uploads a base64 encoded image to Supabase Storage.
    Returns the public URL of the uploaded image.
    """
    try:
        if not image_b64:
            return None
            
        # Decode base64 to bytes
        image_data = base64.b64decode(image_b64)
        
        # Unique filename
        file_name = f"evidence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        file_path = f"alerts/{file_name}"
        
        # Upload to 'incident-evidence' bucket
        supabase.storage.from_("incident-evidence").upload(
            path=file_path,
            file=image_data,
            file_options={"content-type": "image/jpeg"}
        )
        
        # Get public URL
        res = supabase.storage.from_("incident-evidence").get_public_url(file_path)
        return res
    except Exception as e:
        print(f"❌ Supabase Upload Error: {e}")
        return None

def log_incident(audio_class, confidence, decision, reasoning, evidence_url=None):
    """
    Logs incident metadata to the 'incidents' table.
    """
    try:
        data = {
            "audio_class": audio_class,
            "confidence": float(confidence),
            "decision": decision,
            "reasoning": reasoning,
            "evidence_url": evidence_url
        }
        supabase.table("incidents").insert(data).execute()
        print(f"✅ Incident logged to Supabase: {decision}")
    except Exception as e:
        print(f"❌ Supabase Log Error: {e}")

def get_recent_incidents(limit=20):
    """
    Fetches the most recent incidents from the database.
    """
    try:
        res = supabase.table("incidents").select("*").order("created_at", desc=True).limit(limit).execute()
        return res.data
    except Exception as e:
        print(f"❌ Supabase Fetch Error: {e}")
        return []
