from pydantic import BaseModel

#TODO: Use this to properly type throughout the application
class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    query: str
    response: str
    start_time: float
    end_time: float
    elapsed_time: float

class GreetingResponse(BaseModel):
    greeting: str

class LogResponse(BaseModel):
    question: str
    response: str
    elapsed_time: float
    start_time: float
