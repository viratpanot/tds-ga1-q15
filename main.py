from fastapi import FastAPI, Request

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Ledger Agent is running"}

@app.post("/")
async def answer(request: Request):
    body = await request.json()
    question = body.get("question", "")
    return {"answer": f"Received question: {question}"}