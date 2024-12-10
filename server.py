from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
# from .src.model import get_message
from .src.reAct_agent import get_response

# Initialize FastAPI app
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using Langchain's Runnable interfaces",
)

# declare origin/s
origins = [
    "http://localhost:5173",
    "localhost:5173",
    "http://localhost"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

config = {"configurable": {"thread_id": "abc123"}}

class Message(BaseModel):
    msg_data: str

@app.post("/chat")
async def new_message(message: Message):
    # answer = get_message(message.text, config)
    answer = "chat"
    return {"ok": "ok", "answer": answer}

@app.post("/agent")
async def send_message(message: Message):
    try:
        answer = await get_response(message.msg_data, config)
    except Exception as e:
        print(f"There was an exception: {e}")
        raise HTTPException(status_code=404, detail=e)
        # return {"status": "error", "answer": f"There was an exception: {e}"}
    return {"status": "ok", "answer": answer}

# 5. Adding chain route
# add_routes(
#     app,
#     chain,
#     path="/chain",
# )


# Run the server
if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=8000)