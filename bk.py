import requests
import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()  # Загружаем .env

api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ API-ключ не найден! Убедись, что .env файл существует и переменная задана правильно.")

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

data = {
    "model": "deepseek/deepseek-r1:free",
    "messages": [{"role": "user", "content": "Привет!"}]
}

response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)

print(response.status_code)
print(response.text)