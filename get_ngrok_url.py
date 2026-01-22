import requests
try:
    response = requests.get("http://localhost:4040/api/tunnels")
    data = response.json()
    public_url = data['tunnels'][0]['public_url']
    print(f"FULL_NGROK_URL: {public_url}")
except Exception as e:
    print(f"Error: {e}")
