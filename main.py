import json
import re
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

with open("data.json", "r") as f:
    raw_data = json.load(f)

if isinstance(raw_data, list):
    orders = raw_data
elif isinstance(raw_data, dict) and "orders" in raw_data:
    orders = raw_data["orders"]
else:
    raise ValueError("data.json must contain a list of orders or an 'orders' key")

df = pd.json_normalize(orders)

df["updated_at"] = pd.to_datetime(df["updated_at"], errors="coerce", utc=True)

df = (
    df.sort_values("updated_at")
      .drop_duplicates(subset="order_id", keep="last")
)

df["status"] = df["status"].astype(str).str.lower()
df = df[df["status"] == "paid"].copy()

df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

def get_region(question):
    q = question.lower()
    for region in ["north", "south", "east", "west"]:
        if region in q:
            return region
    return None

@app.get("/")
def home():
    return {"message": "Ledger Agent is running"}

@app.post("/")
async def answer(request: Request):
    body = await request.json()
    question = str(body.get("question", ""))
    q = question.lower()

    filtered = df.copy()

    region = get_region(question)
    if region:
        filtered = filtered[
            filtered["region"].astype(str).str.lower() == region
        ]

    if "average" in q or "avg" in q or "mean" in q:
        result = filtered["amount"].mean()
        return JSONResponse({"answer": round(float(result), 2)})

    if "total" in q or "sum" in q or "revenue" in q:
        result = filtered["amount"].sum()
        return JSONResponse({"answer": round(float(result), 2)})

    if "how many" in q or "count" in q or "number of orders" in q:
        return JSONResponse({"answer": int(len(filtered))})

    return JSONResponse({
        "answer": "Unsupported question format"
    })