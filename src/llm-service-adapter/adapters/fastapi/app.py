from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

class Query(BaseModel):
    query: str
    user_id: str

def create_app(adapter):
    app = FastAPI()

    @app.post("/query/")
    async def process_query(query: Query):
        try:
            processed_query, response, timings = adapter.process_query(query.query)
            start_time, end_time, elapsed_time = timings
            adapter.add_response_to_db(
                query.user_id,
                adapter.llm_service.wandb_run.id,
                processed_query,
                response,
                "none",  # No feedback provided
                elapsed_time,
                start_time,
            )
            return {"response": response}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    return app