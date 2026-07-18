import base64 # this is used for encoding and decoding base64 data

# meaning of base64 in simple words: it is a way to convert binary data (like images or files) into a text format that can be easily transmitted over the internet. 
# It takes the binary data and encodes it into a string of ASCII characters, which can then be sent via email, included in URLs, or stored in text files without any issues. When needed, this encoded string can be decoded back into its original binary form.

import requests # this is used to make HTTP requests

# HTTP requests example GET, POST, PUT, DELETE. this is the same https used in https://websites.com 

import io # this is used for handling byte streams
from PIL import Image # this is used for image processing
from dotenv import load_dotenv # used for loading environment variables from a .env file

# environment variables are like secret settings for your application that you don't want to share with everyone.

import os
import logging # used for logging messages for debugging and monitoring

# the word logging means keeping a record of events, messages, or data that happen while a program is running.

#------------------------------------------------------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO) # this sets the minimum threshold (INFO). Msg less important then this are ignored , keeping log output clean
logger = logging.getLogger(__name__) # helps you get source of every msg
# eg: [ERROR] main.py: File Not Found.

load_dotenv()  # Load environment variables from .env file

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Load the API key from environment variables

if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY is not set. Please set it in the .env file.")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - 

def process_image(image_path,query):
    try:
        with open(image_path,'rb') as image_file:
            image_content = image_file.read()
            encoded_image = base64.b64encode(image_content).decode("utf-8")  # utf-8 is a char encoding that converts sequence of char -> bytes or vice-versa
        # now to check whether the image format is right
        try:
            img = Image.open(io.BytesIO(image_content))
            img.verify()
        except Exception as e:
            logger.error(f"Invalid Image format input")
            return {"error" : f"Invalid Image format input"}
        
        # messages is literally how we speak to the model.

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

        # - - - - - - - - - - - - INNER METHOD START - -  - - - - -  - - - - -  - - - - -  - - - - -  

        def make_api_req(model):
            # requests.post() initiates a HTTP POST / asks GROQ server for an answer
            RESPONSE = requests.post(   
                GROQ_API_URL,  # this tells the requests library which server on the internet to connect to.
                json={
                    "model": model,
                    "messages": messages,  # this is ensuring you send the data to GROQ server in JSON string format
                    "max_tokens": 500
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
        
        # - - - - -  - - - - - - - - INNER METHOD END - - - - -  - - - - - - - - - - - - - - -  - - - - 
        
        llama_4Scout_response = make_api_req("meta-llama/llama-4-scout-17b-16e-instruct")
        llama_4Maverick_response = make_api_req("meta-llama/llama-4-maverick-17b-128e-instruct")

        responses = {}
        for model, response in [("LLAMA_4SCOUT", llama_4Scout_response),("LLAMA_4MAVERICK", llama_4Maverick_response)]:
            if response.status_code == 200: # 200 means SUCCESS!Everything worked perfectly.
                result = response.json()
                answer = result["choices"][0]["message"]["content"]  # access one of the responses of several
                logger.info(f"Processed response from {model} API : {answer}")
                responses[model] = answer
            else:
                # Log what went wrong (VERY IMP FOR DEVELOPERS TO DO)
                logger.error(f"{model} failed with status {response.status_code}: {response.text}")
                responses[model] = f"Error: {response.status_code} - {response.text}"
        
        return responses # responses has answers from both models.

    except Exception as e:
        logger.error(f"An unexpected error occured : {str(e)}")
        return {"error" : f"An unexpected error occured : {str(e)}"}

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - -  - - - - -  - - - - -  - - - - -    

if __name__ == "__main__":
    import uvicorn
    # This command starts the web server and tells it to serve your application 'app'
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)