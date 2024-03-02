from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from base64 import b64decode
import io
import subprocess
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

process = subprocess.Popen(["../../build/./digitor", "../../build/digitor.json"],
                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)


@app.post("/upload_image")
async def upload_image(image_data: str = Body(...)):
    dataurl_parts = image_data.split(',')[1]
    decoded_data = b64decode(dataurl_parts)
    try:
        image_stream = io.BytesIO(decoded_data)
        image = Image.open(image_stream)
        pixels = ""
        for pixel in image.getdata():
            pixels += str(pixel[3]) + " "
        process.stdin.write(f"{pixels}\n")
        process.stdin.flush()
        output = process.stdout.readline().strip()
        return {"message": f"{output}"}
    except Exception as e:
        print(f"Error saving image: {e}")
        return {"error": "Failed to save image"}
