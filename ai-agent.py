# %%
# pip install google-genai
# pip install python-dotenv

# %%
import os
from dotenv import load_dotenv

load_dotenv()

# Carrega a API_KEY (insira no arquivo .env)
GEMINI_API = os.environ.get("API_KEY")

# %%
from google import genai

# Gera o cliente
client = genai.Client(api_key=GEMINI_API)

# Gera o modelo
model = "gemini-2.0-flash"

# Lista os modelos disponiveis
for model in client.models.list():
    print(model.name)
    print(model.description)

# %%
response = client.models.generate_content(model=model, contents="Quem foi Dennis Ritchie?")
print(response.text)

# %%
chat = client.chats.create(model=model)
response = chat.send_message("Quem foi Dennis Ritchie?")
print(response.text)

# %%
from google.genai import types

chat_config = types.GenerateContentConfig(
    system_instruction="Você é um assistente de RH com foco em recrutamento e seleção. Você deve analisar currículos e filtrá-los de acordo com as vagas as quais os candidatos estão aplicando."
)

chat = client.chats.create(model=model, config=chat_config)

# %%
prompt = input("Esperando prompt: ")

while prompt != "encerrar":
    response = chat.send_message(prompt)
    print("Resposta: ", response.text)
    print("\n\n")
    prompt = input("Esperando prompt: ")


# %%



