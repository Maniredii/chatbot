from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import google.generativeai as genai
import os
from dotenv import load_dotenv
import base64
from PIL import Image
import io

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure Google AI
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("No Google API key found. Please check your .env file.")

# Initialize Gemini models
genai.configure(api_key=GOOGLE_API_KEY)
text_model = genai.GenerativeModel('gemini-pro')
vision_model = genai.GenerativeModel('gemini-pro-vision')

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories if they don't exist
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str
    option: Optional[str] = None
    image: Optional[str] = None
    image_description: Optional[str] = None

class ChatResponse(BaseModel):
    response: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Enhanced context based on the selected option
        contexts = {
            'technical': """
                You are a friendly technical support specialist.
                Be conversational and helpful.
                Focus on solving technical issues in a clear, step-by-step manner.
                Ask for specific details when needed.
            """,
            'order': """
                You are a friendly order tracking specialist.
                Be conversational and helpful.
                Focus on order status and shipping queries.
                Ask for order numbers when needed.
            """,
            'refund': """
                You are a friendly refunds and returns specialist.
                Be conversational and helpful.
                Focus on guiding customers through the refund process.
                Explain policies clearly and simply.
            """,
            'billing': """
                You are a friendly billing support specialist.
                Be conversational and helpful.
                Focus on resolving payment and billing issues.
                Keep responses clear and secure.
            """
        }

        context = contexts.get(request.option, """
            You are a friendly customer service AI assistant.
            Be conversational and natural in your responses.
            Avoid numbered lists unless specifically needed.
            Keep responses helpful but casual.
            If the customer just says hi or hello, respond naturally and ask how you can help.
        """)

        try:
            if request.image and request.image.strip():  # Only process image if it exists and is not empty
                # Process image
                try:
                    # Decode base64 image
                    image_data = request.image.split(',')[1]
                    image_bytes = base64.b64decode(image_data)
                    
                    # Create image prompt
                    prompt = f"""
                    {context}
                    
                    Customer's Description: {request.image_description or 'No description provided'}
                    Customer's Query: {request.message}
                    
                    Please analyze the image and provide:
                    1. Issue Identification
                    2. Detailed Analysis
                    3. Recommended Solutions
                    """
                    
                    # Generate response using vision model
                    response = vision_model.generate_content([
                        prompt,
                        genai.types.Image(data=image_bytes)
                    ])
                    
                    return ChatResponse(response=response.text)
                except Exception as img_error:
                    print(f"Image processing error: {str(img_error)}")
                    return ChatResponse(response="I had trouble processing the image. Could you describe the issue in text?")
            else:
                # Simplified text-only prompt
                prompt = f"""
                {context}

                Customer: {request.message}

                Respond naturally as a helpful customer service agent. If this is a greeting, 
                welcome the customer and ask how you can help them today.
                """
                
                response = text_model.generate_content(prompt)
                return ChatResponse(response=response.text)
                
        except Exception as e:
            print(f"AI Error: {str(e)}")
            return ChatResponse(response="I encountered an error. Please try again or rephrase your question.")

    except Exception as e:
        print(f"General Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("Starting server...")
    uvicorn.run("app:app", host="127.0.0.1", port=3000, reload=True)
