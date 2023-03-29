# from fake_chat import FakeChat as Chat
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from chat import Chat

# Load environment variables
load_dotenv()

# Initialize the FastAPI app
app = FastAPI()

# Add CORS middleware to the FastAPI app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Create a request model
class ChatRequest(BaseModel):
    query: str


# Create a response model
class ChatResponse(BaseModel):
    response: str


class GreetingResponse(BaseModel):
    greeting: str


# Initialize Chat instance as a global variable
chat = Chat()


@app.get("/")
async def greeting():
    return GreetingResponse(greeting="Hello User! Welcome to the Wandbot API")


# Define the API endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest) -> ChatResponse:
    try:
        print("QUERY: ", chat_request.query)
        response = chat(chat_request.query)
        print("RESPONSE: ", response)
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run the FastAPI app using Uvicorn
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
