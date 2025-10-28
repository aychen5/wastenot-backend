from typing import Optional

from fastapi import FastAPI

app = FastAPI()
# app.py
import os
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client  # pip install supabase
app = FastAPI()

SUPABASE_URL = os.environ["VITE_SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]  # keep server-side only
sb: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# Example EF lookup table in Supabase (table: emission_factors)
# columns: name (pk), landfill_mtco2e_ton, compost_mtco2e_ton

class Item(BaseModel):
    name: str
    leftover_g: float

class Payload(BaseModel):
    session_id: str
    items: list[Item]

def mtco2e_per_ton_to_kg_per_kg(mtco2e):
    # MTCO2e/short ton → kg CO2e/kg
    return (mtco2e * 1000.0) / 907.0

@app.post("/compute")
def compute(payload: Payload, authorization: str | None = Header(None)):
    # (optional) verify Supabase JWT sent from Lovable:
    if not authorization:
        raise HTTPException(401, "Missing auth")
    # Do your own verification if needed, or accept anon.

    # Pull factors in one query
    names = [it.name for it in payload.items]
    factors = sb.table("emission_factors").select("*").in_("name", names).execute().data
    factor_map = {r["name"]: r for r in factors}

    total_landfill = total_compost = 0.0
    item_results = []

    for it in payload.items:
        f = factor_map.get(it.name)
        if not f:
            continue
        landfill_factor = mtco2e_per_ton_to_kg_per_kg(f["landfill_mtco2e_ton"])
        compost_factor  = mtco2e_per_ton_to_kg_per_kg(f["compost_mtco2e_ton"])
        kg = it.leftover_g / 1000.0
        lf = kg * landfill_factor
        cf = kg * compost_factor
        item_results.append({
            "name": it.name,
            "leftover_g": it.leftover_g,
            "landfill_kgco2e": lf,
            "compost_kgco2e": cf,
            "avoided_kgco2e": lf - cf
        })
        total_landfill += lf; total_compost += cf

    out = {
        "session_id": payload.session_id,
        "items": item_results,
        "totals": {
            "landfill_kgco2e": total_landfill,
            "compost_kgco2e": total_compost,
            "avoided_kgco2e": total_landfill - total_compost
        }
    }

    # (optional) persist a result row
    sb.table("results").insert({
        "session_id": payload.session_id,
        "totals": out["totals"],
        "items": item_results
    }).execute()

    return out


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}