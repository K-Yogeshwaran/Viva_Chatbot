from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Initialize FastAPI app
app = FastAPI()

# Load the Llama 2 model (replace with your specific model)
model = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf")

# Define request schema
class QueryRequest(BaseModel):
    query: str

# API endpoint to process user queries
@app.post("/get_response/")
async def get_response(request: QueryRequest):
    user_input = request.query
    response = model(user_input, max_length=150, do_sample=True)[0]["generated_text"]
    return {"response": response}

# Run the API using: uvicorn app:app --host 0.0.0.0 --port 8000
