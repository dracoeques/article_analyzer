import requests

def openai_apikey_info(apikey: str):
    response = requests.get('https://api.openai.com/v1/files', headers={'Authorization': f"Bearer {apikey}"})
    return response.headers['Openai-Ratelimit-Remaining']
    # for header, value in response.headers.items():
    #     print(header + ": " + value)
