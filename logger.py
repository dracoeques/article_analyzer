import requests
import os
from dotenv import load_dotenv, find_dotenv

class Logger:
    dscd_url: str
    dscd_headers: dict[str: str]

    def __init__(self):
        load_dotenv(find_dotenv())
        token = os.environ["DISCORD_TOKEN"]
        channel_id = os.environ["DISCORD_CHANNEL_ID"]
        self.dscd_url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
        self.dscd_headers = {
            "Authorization": f"Bot {token}",
            "Content-Type": "application/json"
        }   
    
    def log(self, content):
        mode = os.environ["MODE"]
        try:
            if mode == "dev":
                print(content)
                data = {
                    "content": content
                }
                response = requests.post(self.dscd_url, headers=self.dscd_headers, json=data)
                if response.status_code != 200:
                    print(f"Failed to send discord message. Status code: {response.status_code}")
                    print(response.json())
        
            elif mode == "prod":
                data = {
                    "content": content
                }
                response = requests.post(self.dscd_url, headers=self.dscd_headers, json=data)
                if response.status_code != 200:
                    print(f"Failed to send discord message. Status code: {response.status_code}")
                    print(response.json())
        except Exception as er:
            print(er)