import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.environ["LEDGER_URL"]

response = requests.get(BASE_URL, timeout=30)
response.raise_for_status()

data = response.json()

print(json.dumps(data, indent=2))