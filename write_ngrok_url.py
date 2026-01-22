import requests
with open("ngrok_public_url.txt", "w") as f:
    try:
        response = requests.get("http://localhost:4040/api/tunnels")
        data = response.json()
        public_url = data['tunnels'][0]['public_url']
        f.write(public_url)
    except Exception as e:
        f.write(f"Error: {e}")
