import json
import traceback
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

def load_orders():
    with open("data.json", "r") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        rows = raw
    elif isinstance(raw, dict):
        for possible_key in ["orders", "data", "results", "items"]:
            if possible_key in raw and isinstance(raw[possible_key], list):
                rows = raw[possible_key]
                break
        else:
            raise ValueError(
                f"Could not find an orders list. Top-level keys are: {list(raw.keys())}"
            )
    else:
        raise ValueError("data.json must be a JSON list or object")

    return pd.json_normalize(rows)

df = load_orders()

@app.get("/")
def home():
    return {
        "message": "Ledger Agent is running",
        "rows_loaded": len(df)
    }

@app.get("/debug")
def debug():
    return {
        "rows_loaded": len(df),
        "columns": df.columns.tolist(),
        "first_three_rows": df.head(3).fillna("").to_dict(orient="records")
    }

def find_column(candidates):
    lower_to_actual = {str(col).lower(): col for col in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower_to_actual:
            return lower_to_actual[candidate.lower()]

    for actual_lower, actual_name in lower_to_actual.items():
        for candidate in candidates:
            if candidate.lower() in actual_lower:
                return actual_name

    return None

def prepared_orders():
    work = df.copy()

    order_id_col = find_column(["order_id", "orderid", "id"])
    updated_col = find_column(["updated_at", "updatedat", "last_updated", "modified_at"])
    status_col = find_column(["status", "order_status"])

    if order_id_col and updated_col:
        work[updated_col] = pd.to_datetime(
            work[updated_col], errors="coerce", utc=True
        )
        work = (
            work.sort_values(updated_col)
                .drop_duplicates(subset=order_id_col, keep="last")
        )

    if status_col:
        work = work[
            work[status_col].astype(str).str.strip().str.lower() == "paid"
        ].copy()

    return work

@app.post("/")
async def answer(request: Request):
    try:
        body = await request.json()
        question = str(body.get("question", ""))
        q = question.lower()

        work = prepared_orders()

        region_col = find_column(["region", "sales_region", "customer_region"])
        amount_col = find_column([
            "amount_usd",
            "total_usd",
            "revenue_usd",
            "amount",
            "total",
            "revenue",
            "order_value",
            "value"
        ])

        if not amount_col:
            return JSONResponse({
                "answer": "DATA ERROR: Could not find the money column. Visit /debug and inspect columns."
            })

        work[amount_col] = pd.to_numeric(work[amount_col], errors="coerce")

        for region in ["north", "south", "east", "west"]:
            if region in q:
                if not region_col:
                    return JSONResponse({
                        "answer": "DATA ERROR: Could not find the region column. Visit /debug and inspect columns."
                    })

                work = work[
                    work[region_col].astype(str).str.strip().str.lower() == region
                ]

        if "average" in q or "avg" in q or "mean" in q:
            value = work[amount_col].mean()
            if pd.isna(value):
                return JSONResponse({"answer": 0})
            return JSONResponse({"answer": round(float(value), 2)})

        if "how many" in q or "count" in q or "number of" in q:
            return JSONResponse({"answer": int(len(work))})

        if "total" in q or "sum" in q or "revenue" in q:
            value = work[amount_col].sum()
            return JSONResponse({"answer": round(float(value), 2)})

        return JSONResponse({
            "answer": "Unsupported question wording"
        })

    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse({
            "answer": f"SERVER ERROR: {type(e).__name__}: {str(e)}"
        })