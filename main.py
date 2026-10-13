import os, re, json, threading, urllib.parse, urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE = re.sub(r"&(?:path|page)=[^&]*", "", os.environ["LEDGER_BASE"].strip().strip("'\""))          # URL up to ...&questionId=q-ledger-agent-server
IST = timezone(timedelta(hours=5, minutes=30))
DATE_FIELD = "created_at"                 # timestamp that defines the business date
CENT = Decimal("0.01")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------- data loading ----------

def fetch(path, raw=False, **params):
    u = f"{BASE}&path={urllib.parse.quote(path, safe='')}"
    if params:
        u += "&" + urllib.parse.urlencode(params)
    req = urllib.request.Request(u, headers={
        "User-Agent": "Mozilla/5.0 (compatible; ledger-agent/1.0)",
        "Accept": "application/json",
    })
    with urllib.request.urlopen(req, timeout=12) as r:
        body = r.read().decode()
    return body if raw else json.loads(body)


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))

_lock, _data = threading.Lock(), None

def _load():
    rates = fetch("/rates")["usd_per_unit"]
    first = fetch("/orders", page=1)
    rows = list(first["orders"])
    if first["pages"] > 1:
        with ThreadPoolExecutor(8) as ex:
            for page in ex.map(lambda p: fetch("/orders", page=p)["orders"],
                               range(2, first["pages"] + 1)):
                rows.extend(page)
    latest = {}
    for r in rows:                                   # newest updated_at wins
        cur = latest.get(r["id"])
        if cur is None or ts(r["updated_at"]) >= ts(cur["updated_at"]):
            latest[r["id"]] = r
    recs = [{
        "id": r["id"], "status": str(r["status"]).strip().lower(),
        "region": r["region"], "product": r["product"], "customer": r["customer"],
        "qty": int(r["qty"]),
        "usd": Decimal(str(r["amount"])) * Decimal(str(rates[r["currency"]])),
        "day": ts(r[DATE_FIELD]).astimezone(IST).date(),
    } for r in latest.values()]
    return {
        "rows": len(rows), "recs": recs,
        "regions": sorted({r["region"] for r in recs}),
        "products": sorted({r["product"] for r in recs}),
        "statuses": sorted({r["status"] for r in recs}),
        "year": Counter(r["day"].year for r in recs).most_common(1)[0][0],
    }

def get_data():
    global _data
    with _lock:
        if _data is None:
            _data = _load()
    return _data

threading.Thread(target=lambda: get_data(), daemon=True).start()   # warm up at boot


# ---------- date parsing ----------
MONTHS = {m: i + 1 for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split())}
MON = (r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
       r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")

def month_span(y, m):
    return date(y, m, 1), date(y + (m == 12), m % 12 + 1, 1) - timedelta(days=1)

def parse_period(q, dy):
    days = [date(int(a), int(b), int(c)) for a, b, c in re.findall(r"\b(\d{4})-(\d{2})-(\d{2})\b", q)]
    for mo, d, y in re.findall(rf"\b{MON}\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", q):
        days.append(date(int(y), MONTHS[mo[:3]], int(d)))
    for d, mo, y in re.findall(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?{MON}\.?,?\s+(\d{{4}})\b", q):
        days.append(date(int(y), MONTHS[mo[:3]], int(d)))
    if days:
        return min(days), max(days)
    ym = re.search(r"\b(20\d{2})\b", q)
    gy = int(ym.group(1)) if ym else dy
    qm = re.search(r"\bq([1-4])\b|\b(first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter\b", q)
    if qm:
        n = int(qm.group(1)) if qm.group(1) else {"first": 1, "1st": 1, "second": 2, "2nd": 2,
                                                  "third": 3, "3rd": 3, "fourth": 4, "4th": 4}[qm.group(2)]
        return date(gy, 3 * n - 2, 1), month_span(gy, 3 * n)[1]
    ments = [[int(m), int(y)] for y, m in re.findall(r"\b(20\d{2})-(0[1-9]|1[0-2])\b", q)]
    for mo, y in re.findall(rf"\b{MON}\b\.?(?:,?\s*(?:of\s+)?(20\d{{2}}))?", q):
        ments.append([MONTHS[mo[:3]], int(y) if y else None])
    nxt = None
    for mt in reversed(ments):
        if mt[1] is None:
            mt[1] = nxt or gy
        nxt = mt[1]
    if ments:
        spans = [month_span(y, m) for m, y in ments]
        return min(s for s, _ in spans), max(e for _, e in spans)
    if ym:
        y = int(ym.group(1))
        return date(y, 1, 1), date(y, 12, 31)
    return None, None


# ---------- question answering ----------
NUMW = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "ten": 10}

def agg(rs, measure):
    if measure == "money":
        return sum((r["usd"] for r in rs), Decimal(0))
    if measure == "units":
        return sum(r["qty"] for r in rs)
    if measure == "orders":
        return len(rs)
    if measure == "customers":
        return len({r["customer"] for r in rs})
    if measure == "avg":
        return sum((r["usd"] for r in rs), Decimal(0)) / len(rs) if rs else Decimal(0)

def fmt(v, measure):
    if measure in ("money", "avg"):
        return float(Decimal(v).quantize(CENT, rounding=ROUND_HALF_UP))
    return int(v)

def answer(question):
    d = get_data()
    recs = d["recs"]
    q = re.sub(r"\s+", " ", question.lower())

    # single order lookup
    m = re.search(r"\bord-\d+\b", q)
    if m:
        r = next((x for x in recs if x["id"].lower() == m.group(0)), None)
        if r is None:
            return None
        if "status" in q:
            return r["status"]
        if re.search(r"\b(units?|quantity|qty)\b", q):
            return r["qty"]
        return fmt(r["usd"], "money")

    regions = [x for x in d["regions"] if re.search(rf"\b{re.escape(x.lower())}\b", q)]
    products = [p for p in d["products"]
                if re.search(rf"\b{re.sub(r'[ -]+', r'[ -]+', re.escape(p.lower()).replace(chr(92) + ' ', ' '))}(?:s|es)?\b", q)]
    custs = {c.upper() for c in re.findall(r"\bc\d{3,6}\b", q)}
    start, end = parse_period(q, d["year"])

    refund = bool(re.search(r"\brefund|\breturned\b", q))
    if refund:
        statuses = {"refunded"}
    elif re.search(r"\bvoid(ed)?\b|\bcancel", q):
        statuses = {"void"}
    elif re.search(r"all orders|any status|regardless of status|every order", q):
        statuses = None
    else:
        statuses = {"paid"}

    def select(sts):
        out = []
        for r in recs:
            if sts is not None and r["status"] not in sts: continue
            if regions and r["region"] not in regions: continue
            if products and r["product"] not in products: continue
            if custs and r["customer"] not in custs: continue
            if start and not (start <= r["day"] <= end): continue
            out.append(r)
        return out

    rs = select(statuses)

    # ranking questions: "which region ...", "top 3 products ..."
    sup = re.search(r"\b(top|best[- ]?selling|best|highest|most|largest|biggest|greatest|leading|"
                    r"lowest|least|worst|fewest|smallest|bottom)\b", q)
    dim = None
    if sup:
        for c in re.findall(r"\b(region|product|customer)s?\b", q):
            if (c == "region" and regions) or (c == "product" and products) or (c == "customer" and custs):
                continue
            dim = c
            break
    if dim:
        if re.search(r"\b(units?|quantity|qty|pieces|sold)\b", q):
            measure = "units"
        elif re.search(r"\borders?\b", q) and not re.search(r"revenue|sales|spen[dt]|amount|value|refund", q):
            measure = "orders"
        else:
            measure = "money"
        nm = re.search(r"\b(?:top|bottom)\s+(\d+|one|two|three|four|five|ten)\b", q)
        n = int(NUMW.get(nm.group(1), nm.group(1))) if nm else 1
        asc = bool(re.search(r"\b(lowest|least|worst|fewest|smallest|bottom)\b", q))
        key = {"region": "region", "product": "product", "customer": "customer"}[dim]
        groups = defaultdict(list)
        for r in rs:
            groups[r[key]].append(r)
        scores = {k: agg(v, measure) for k, v in groups.items()}
        order = sorted(scores, key=lambda k: (scores[k] if asc else -scores[k], k))
        if not order:
            return None
        return order[0] if n == 1 else order[:n]

    # scalar questions
    if re.search(r"refund rate|rate of refund", q):
        rs_all = select(None)
        return round(100 * len([r for r in rs_all if r["status"] == "refunded"]) / len(rs_all), 2) if rs_all else 0
    if re.search(r"(how many|number of|count of)\s+(of\s+)?(unique |distinct |different )?customers|"
                 r"(unique|distinct) customers", q):
        measure = "customers"
    elif re.search(r"\b(units?|quantity|qty|pieces)\b", q):
        measure = "units"
    elif re.search(r"average|\bavg\b|\bmean\b|\baov\b", q):
        measure = "avg"
    elif re.search(r"how many|number of|count of", q):
        measure = "orders"
    else:
        measure = "money"
    return fmt(agg(rs, measure), measure)


class Q(BaseModel):
    question: str

@app.post("/")
def ask(body: Q):
    try:
        return {"answer": answer(body.question)}
    except Exception as e:                      # never time out silently
        return {"answer": None, "error": str(e)}

@app.get("/debug")
def debug():
    d = get_data()
    return {"rows_fetched": d["rows"], "unique_orders": len(d["recs"]),
            "export_lines": len(fetch("/export", raw=True).splitlines()),
            "statuses": d["statuses"], "regions": d["regions"], "products": d["products"]}