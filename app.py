from fastapi import FastAPI, File , UploadFile , HTTPException , Request , Form
from fastapi.templating import Jinja2Templates # what jina does is to render html files , render is to display the html file
from fastapi.responses import HTMLResponse,JSONResponse # for returning html response 

# --------------------------------- COPIED FROM main.py -----------------------------------------------------------------------------------------------

import base64 
import requests
import io
from PIL import Image 
from dotenv import load_dotenv 
import os
import logging

logging.basicConfig(level=logging.INFO) # this sets the minimum threshold (INFO). Msg less important then this are ignored , keeping log output clean
logger = logging.getLogger(__name__) # helps you get source of every msg
# eg: [ERROR] main.py: File Not Found.

load_dotenv()  # Load environment variables from .env file


app = FastAPI() # create an instance of FastAPI , FastAPI is a web framework for building APIs with Python.
templates = Jinja2Templates(directory="templates") 

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Load the API key from environment variables

if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY is not set. Please set it in the .env file.")


# ---------------------------------- COPIED FROM main.py ------------------------------------------------------------------------------------------

# raise -> to throw an error when something goes wrong

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request, "index.html", {})

# our response_class is HTMLResponse because we want FastAPI to return an html file not JSON [default]. since its a website/webpage.
# returning as JSON would show user raw HTML code instead of rendering the webpage.
# async def is a function that can run multiple tasks at the same time without having to wait for each task to complete [pipeline analogy]

@app.post('/upload_and_query') # route to post image and query 
async def upload_and_query(image: UploadFile = File(...), query: str = Form(...)):
    try:
        image_content = await image.read()  # Read the uploaded image file
        if not image_content:
            raise HTTPException(status_code=400, detail="No image uploaded") # meaning of 400 is bad request
        
        encoded_image = base64.b64encode(image_content).decode('utf-8') # encode image to base64 string

        # - - - - - - - - - - CHECK IMG FORMAT START  - - - - - - - - - -
        try:
            img = Image.open(io.BytesIO(image_content))
            img.verify()
        except Exception as e:
            logger.error(f"Invalid Image format input")
            raise HTTPException(status_code=400, detail="Invalid Image format input")
        # - - - - - - - - - - CHECK IMG FORMAT END - - - - - - - - - 
        # - - - - -  - - - - - SAME CODE AS main.py - - - - - - - -

        messages = [
            {
                "role": "user", # speaker of the msg is -> user (me)
                "content": [
                    {"type": "text", # user will type his question
                     "text": query
                    },
                    {"type": "image_url", # provide a image link related to question
                     "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
                    }
                ]
            }
        ]

        def make_api_req(model):
            # requests.post() initiates a HTTP POST / asks GROQ server for an answer.
            RESPONSE = requests.post(   
                GROQ_API_URL,  # this tells the requests library which server on the internet to connect to.
                json={
                    "model": model,
                    "messages": messages,  # this is ensuring you send the data to GROQ server in JSON string format
                    "max_tokens": 500,
                    "reasoning_effort": "none"
                },
                headers = {
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                timeout = 30
                # JSON is the universal format for data exchange between web services
                # we use it for simplicity and standardization
            )
            return RESPONSE
        
        qwen_response = make_api_req("qwen/qwen3.6-27b")

        responses = {}
        for model, response in [("QWEN36", qwen_response)]:
            if response.status_code == 200: # 200 means SUCCESS!Everything worked perfectly.
                result = response.json()
                answer = result["choices"][0]["message"]["content"]  # access one of the responses of several
                logger.info(f"Processed response from {model} API : {answer}")
                responses[model] = answer
            else:
                # Log what went wrong (VERY IMP FOR DEVELOPERS TO DO)
                logger.error(f"{model} failed with status {response.status_code}: {response.text}")
                responses[model] = f"Error: {response.status_code} - {response.text}"
        return JSONResponse(status_code = 200 , content=responses) 
    
        # JSON response because frontend JS will parse this JSON to display answers from both models on webpage.
    
        # - - - - -  - - - - - - - -SAME CODE AS main.py - - - - - - - 
    except HTTPException as he:
        logger.error(f"HTTP exception: {str(he)}")
        raise he
    except Exception as e:
        logger.error(f"Error reading image file: {str(e)}")
        raise HTTPException(status_code=400, detail = f"An unexpected error occured : {str(e)}")
    
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,port=8000)

# uvicorn is an ASGI server for running FastAPI applications
# ASGI server is a server that can handle asynchronous web applications, allowing for better performance and scalability.
# ASGI stands for Asynchronous Server Gateway Interface
