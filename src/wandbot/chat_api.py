from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from chat import Chat
# from fake_chat import FakeChat as Chat
from wandbot.config import TEAM, PROJECT, JOB_TYPE, default_config
import wandb


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
    query: str
    response: str
    start_time: float
    end_time: float
    elapsed_time: float


# Create a response model
class GreetingResponse(BaseModel):
    greeting: str

# Define the chatbot and wandb run for each worker
wandb_run = None
chat = None

# We will use this to properly initialize the chatbot and wandb run for each worker
@app.on_event("startup")
async def startup_event():
  global wandb_run, chat
  wandb_run = wandb.init(
      entity=TEAM,
      project=PROJECT,
      job_type=JOB_TYPE,
      config=default_config,
  )
  chat = Chat(model_name=wandb_run.config.model_name, wandb_run=wandb_run)

@app.get("/")
async def greeting():
    return GreetingResponse(greeting="Hello User! Welcome to the WandBot API")

# Define the API endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest) -> ChatResponse:
    global chat
    try:
        print("QUERY: ", chat_request.query)
        query, response, timings = chat(chat_request.query)
        start_time, end_time, elapsed_time = timings
        print("TIMINGS: ", timings)
        print("RESPONSE: ", response)
        return ChatResponse(
            query=query,
            response=response,
            start_time=start_time,
            end_time=end_time,
            elapsed_time=elapsed_time,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/flush_tracker")
async def flush_tracker():
    global wandb_run, chat
    try:
        wandb_run.finish()
        wandb_run = wandb.init(
            entity=TEAM,
            project=PROJECT,
            job_type=JOB_TYPE,
            config=default_config,
        )
        chat.wandb_run = wandb_run
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("shutdown")
async def shutdown_event():
  global wandb_run
  wandb_run.finish()

# Run the FastAPI app using Uvicorn
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
