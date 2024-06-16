# pip install pytorch transformers datasets langchain python-multipart fastapi uvicorn langchain-community pyngrok
# !curl -fsSL https://ollama.com/install.sh | sh

# uvicorn BackendHackathon:app --reload

from langchain_community.llms import Ollama
import base64
from typing import Dict
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from fastapi import FastAPI , HTTPException, UploadFile,File
from fastapi.responses import RedirectResponse
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
import requests
import torch
from PIL import Image
from transformers import *
# from tqdm import tqdm
import urllib.parse as parse
import json
# import nest_asyncio
# from pyngrok import ngrok
import os

import subprocess
# subprocess.Popen("ollama serve",shell=True)
# subprocess.Popen("ollama run llava:v1.6",shell=True)

# subprocess.Popen("ollama pull llava:v1.6", shell=True)
subprocess.Popen("ollama serve", shell=True)
# subprocess.Popen("ollama run llava:v1.6", shell=True)
import time
# time.sleep(100)
llava = Ollama(model="llava:v1.6")
print(llava("Hi"))

app = FastAPI()
origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

def image_to_base64(image_path):
    if image_path.startswith('http://') or image_path.startswith('https://'):
        response = requests.get(image_path)
        if response.status_code == 200:
            encoded_string = base64.b64encode(response.content)
            return encoded_string.decode("utf-8")
        else:
            print("Failed to fetch image from URL:", image_path)
            return None
    else:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
            return encoded_string.decode("utf-8")

device = "cuda" if torch.cuda.is_available() else "cpu"

  # a function to determine whether a string is a URL or not
def is_url(string):
    try:
        result = parse.urlparse(string)
        return all([result.scheme, result.netloc, result.path])
    except:
        return False

# a function to load an image
def load_image(image_path):
    if is_url(image_path):
        return Image.open(requests.get(image_path, stream=True).raw)
    elif os.path.exists(image_path):
        return Image.open(image_path)

@app.post("/caption1")
async def UploadImage(file: bytes = File(...)):
  with open('image.jpg','wb') as image:
    image.write(file)
    image.close()
  title = llava(prompt="If the provided image is clear and of good quality, generate a vivid, engaging title that captures the essence of the image. The title should be short and to the point, giving a general view of the main scene depicted without mentioning quantities or numbers. Focus on providing an impactful description of the primary subject or action in the image. However, if the image is blurred, distorted, or of poor quality, respond by saying 'The image quality is too low to generate an accurate title.",images=[str(image_to_base64(str("/content/image.jpg")))])
  return {"title":title}

@app.post("/caption")
async def UploadImage(file: bytes = File(...)):
  with open('image.jpg','wb') as image:
    image.write(file)
    image.close()

  params = {
    'models': 'quality',
    'api_user': '1276533234',
    'api_secret': 'fLqocaoDSMvC7BvwdFZbeJbfmq3KP4Gk'
  }

  files = {'media': open('./image.jpg', 'rb')}
  r = requests.post('https://api.sightengine.com/1.0/check.json', files=files, data=params)

  score = json.loads(r.text)['quality']['score']
  print(score)
  if score <= 0.45:
    return {"title":"Quality issues are too severe to recognize visual content."}

  title = llava(prompt="Generate a vivid, engaging title that captures the essence of the provided image. The title should be short and to the point that give a general view of the image. Do not mention quantities or numbers. Focus on providing an impactful description of the main scene depicted.",images=[str(image_to_base64(str("./image.jpg")))])
  return {"title":title}

@app.post("/chat")
async def Chat(prompt:str,file: bytes = File(...)):
  with open('image.jpg','wb') as image:
    image.write(file)
    image.close()
  RES = llava(prompt=prompt,images=[str(image_to_base64(str("./image.jpg")))]  )
  return {"content":RES}

from huggingface_hub import login
import os
os.environ["HF_TOKEN"] = "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"



# """# Using a Trained Model"""

# model = VisionEncoderDecoderModel.from_pretrained("Stormbreakerr20/vit-gpt2-vizwiz").to(device)
# tokenizer = GPT2TokenizerFast.from_pretrained("Stormbreakerr20/vit-gpt2-vizwiz")
# image_processor = ViTImageProcessor.from_pretrained("Stormbreakerr20/vit-gpt2-vizwiz")
# torch.save(model.state_dict(), "model.pth")
# a function to perform inference

# @app.post("/captions")
# def UploadImage(file: bytes = File(...)):
#     with open('image.jpg','wb') as image:
#       image.write(file)
#       image.close()

#     params = {
#     'models': 'quality',
#     'api_user': '1276533234',
#     'api_secret': 'fLqocaoDSMvC7BvwdFZbeJbfmq3KP4Gk'
#     }
#     files = {'media': open('/content/image.jpg', 'rb')}
#     r = requests.post('https://api.sightengine.com/1.0/check.json', files=files, data=params)

#     score = json.loads(r.text)['quality']['score']
#     print(score)
#     if score <= 0.45:
#       return {"title":"Quality issues are too severe to recognize visual content."}

#     image = Image.open('/content/image.jpg')
#     img = image_processor(image, return_tensors="pt").to(device)
#     model.to(device)
#     with torch.no_grad():
#         output = model.generate(**img)
#     caption = tokenizer.batch_decode(output, skip_special_tokens=True)[0]


#     title = llava(prompt=f"here is the summary of the image provided: {caption}, Based on the given summary and image analyse both of these and Generate a vivid, engaging title that captures the essence of the provided image. The title should be short and to the point that give a general view of the image.",images=[str(image_to_base64(str("/content/image.jpg")))])
#     return {"title":title}

# Get your authtoken from https://dashboard.ngrok.com/get-started/your-authtoken
# os.system(f"ngrok authtoken 2fX4X2BATxalJmNiCutMXV1qR6k_4RXntzTaGfFuGwYpRuKcE")

# # Connect to ngrok
# connection_string = ngrok.connect(8000,bind_tls = True)

# # Print the public URL
# print('Public URL:', connection_string)
# nest_asyncio.apply()
# uvicorn.run(app,port = 8000,reload = False)