# app.py
import os
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from supabase import create_client, Client


SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]  # keep server-side only
sb: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


sb: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# ----- FastAPI app + CORS -----
app = FastAPI(title="WasteNotNYC Emissions API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "https://a74f6327-764b-47f2-8e5a-0cf826e093a1.lovable.app",
    "http://localhost:3000",  # for local testing
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----- Schemas -----
class Component(BaseModel):
    name: str
    leftover_g: float = Field(..., ge=0, description="grams of leftover for this component")

class Payload(BaseModel):
    userId: Optional[str] = None
    session_id: Optional[str] = None
    components: List[Component] = Field(..., min_items=1)

class ItemResult(BaseModel):
    name: str
    leftover_g: float
    landfill_kgco2e: float
    compost_kgco2e: float
    avoided_kgco2e: float

class Totals(BaseModel):
    landfill_kgco2e: float
    compost_kgco2e: float
    avoided_kgco2e: float

class ComputeResponse(BaseModel):
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    items: List[ItemResult]
    totals: Totals
    db_row_id: Optional[str] = None
    db_error: Optional[str] = None

# ----- Helpers -----
def mtco2e_per_ton_to_kg_per_kg(mtco2e_per_short_ton: float) -> float:
    # WARM factors are MTCO2e per short ton; convert to kg CO2e per kg
    # 1 short ton ≈ 907 kg; 1 MT = 1000 kg
    return (float(mtco2e_per_short_ton) * 1000.0) / 907.0

_factor_cache: Dict[str, Dict] = {}

def preload_factors():
    """Load all factors once into memory (safe if called multiple times)."""
    global _factor_cache
    if _factor_cache:
        return
    res = sb.table("emission_factors").select(
        "name,landfill_mtco2e_ton,compost_mtco2e_ton"
    ).execute()
    for r in (res.data or []):
        _factor_cache[r["name"]] = r

def get_factor(name: str) -> Optional[Dict]:
    """Exact match, then generic fallbacks."""
    if not _factor_cache:
        preload_factors()
    return (
        _factor_cache.get(name)
        or _factor_cache.get("Food Waste (non-meat)")
        or _factor_cache.get("Food Waste")
    )

# ----- Core endpoint -----
@app.post("/calculate-emissions", response_model=ComputeResponse)
def calculate_emissions(data: Payload):
    preload_factors()  # ensure cache is ready

    items_out: List[ItemResult] = []
    total_landfill = 0.0
    total_compost = 0.0

    for comp in data.components:
        f = get_factor(comp.name)
        if not f:
            # no factor found anywhere; record a zero row
            items_out.append(ItemResult(
                name=comp.name, leftover_g=comp.leftover_g,
                landfill_kgco2e=0.0, compost_kgco2e=0.0, avoided_kgco2e=0.0
            ))
            continue

        landfill_factor = mtco2e_per_ton_to_kg_per_kg(f["landfill_mtco2e_ton"])
        compost_factor  = mtco2e_per_ton_to_kg_per_kg(f["compost_mtco2e_ton"])

        kg = comp.leftover_g / 1000.0
        landfill = kg * landfill_factor
        compost  = kg * compost_factor
        avoided  = landfill - compost

        items_out.append(ItemResult(
            name=comp.name,
            leftover_g=comp.leftover_g,
            landfill_kgco2e=round(landfill, 6),
            compost_kgco2e=round(compost, 6),
            avoided_kgco2e=round(avoided, 6),
        ))

        total_landfill += landfill
        total_compost  += compost

    totals = Totals(
        landfill_kgco2e=round(total_landfill, 6),
        compost_kgco2e=round(total_compost, 6),
        avoided_kgco2e=round(total_landfill - total_compost, 6),
    )

    db_error = None
    db_row_id = None
    try:
        db_row = sb.table("emissions_results").insert({
            "user_id": data.userId,
            "session_id": data.session_id,
            "items": [i.model_dump() for i in items_out],
            "totals": totals.model_dump(),
            "source": "lovable",
            "warm_version": "v16"
        }).execute()
        if db_row.data:
            db_row_id = db_row.data[0].get("id")
    except Exception as e:
        # don't fail the request; return results and an error string
        db_error = str(e)

    return ComputeResponse(
        user_id=data.userId,
        session_id=data.session_id,
        items=items_out,
        totals=totals,
        db_row_id=db_row_id,
        db_error=db_error
    )

# ----- Health & root -----
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/")
def root():
    return {"ok": True, "service": "WasteNotNYC Emissions API"}