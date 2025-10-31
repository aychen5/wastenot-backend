# app.py
import os
import re
import logging
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from supabase import create_client, Client

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]  # keep server-side only
sb: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# ----- FastAPI app + CORS -----
app = FastAPI(title="WasteNotNYC Emissions API", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://.*\.lovable\.app",
    allow_origins=[
        "https://a74f6327-764b-47f2-8e5a-0cf826e093a1.lovable.app",
        "http://localhost:3000",  # for local testing
        "http://localhost:5173",  # common Vite dev server port
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# Helper function to check if origin is allowed
def is_allowed_origin(origin: Optional[str]) -> bool:
    """Check if origin is in allowed list."""
    if not origin:
        return False
    allowed_origins = [
        "https://a74f6327-764b-47f2-8e5a-0cf826e093a1.lovable.app",
        "http://localhost:3000",
        "http://localhost:5173",
    ]
    if origin in allowed_origins:
        return True
    # Check regex pattern
    pattern = r"https://.*\.lovable\.app"
    return bool(re.match(pattern, origin))

# Exception handlers to ensure CORS headers are added to all errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with CORS headers."""
    origin = request.headers.get("origin")
    cors_origin = origin if is_allowed_origin(origin) else None
    headers = {}
    if cors_origin:
        headers = {
            "Access-Control-Allow-Origin": cors_origin,
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, HEAD",
            "Access-Control-Allow-Headers": "*",
        }
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body},
        headers=headers
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with CORS headers."""
    origin = request.headers.get("origin")
    cors_origin = origin if is_allowed_origin(origin) else None
    headers = {}
    if cors_origin:
        headers = {
            "Access-Control-Allow-Origin": cors_origin,
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, HEAD",
            "Access-Control-Allow-Headers": "*",
        }
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=headers
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions with CORS headers."""
    origin = request.headers.get("origin")
    cors_origin = origin if is_allowed_origin(origin) else None
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    headers = {}
    if cors_origin:
        headers = {
            "Access-Control-Allow-Origin": cors_origin,
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, HEAD",
            "Access-Control-Allow-Headers": "*",
        }
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
        headers=headers
    )


# ----- Schemas -----
class Component(BaseModel):
    name: str
    leftover_g: float = Field(..., ge=0, description="grams of leftover for this component")

class Payload(BaseModel):
    userId: Optional[str] = None
    session_id: Optional[str] = None
    components: List[Component] = Field(..., min_items=1)
    selected_items: Optional[List[str]] = Field(None, description="List of item names selected from items table (e.g., ['Garden Salad', 'Tea'])")
    packaging_item_ids: Optional[List[str]] = Field(None, description="List of packaging item IDs from packaging_items table")


class ItemResult(BaseModel):
    component_name: str
    category: str
    share: float
    component_leftover_g: float
    landfill_kgco2e: float
    compost_kgco2e: float
    avoided_kgco2e: float
    landfill_factor_kg_per_kg: float
    compost_factor_kg_per_kg: float

class Totals(BaseModel):
    landfill_kgco2e: float
    compost_kgco2e: float
    avoided_kgco2e: float
    avoided_kgco2e_food: float
    avoided_kgco2e_packaging: float

class ComputeResponse(BaseModel):
    session_id: Optional[str] = None
    meal_name: Optional[List[str]] = None
    leftover_g: float
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
        "name,landfill_mtco2e_ton,compost_mtco2e_ton,recycle_mtco2e_ton,leftover_percentage"
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

def load_packaging_items(packaging_item_ids: Optional[List[str]]) -> List[Dict]:
    """Load packaging items from packaging_items table by IDs."""
    if not packaging_item_ids:
        return []
    try:
        res = sb.table("packaging_items").select(
            "id,name,grams,emission_factor_category"
        ).in_("id", packaging_item_ids).execute()
        return res.data or []
    except Exception as e:
        logger.error(f"Error loading packaging items: {e}", exc_info=True)
        return []


class MealEmissionsRequest(BaseModel):
    item_id: Optional[str] = None
    item_name: Optional[str] = None
    session_id: Optional[str] = None
    userId: Optional[str] = None
    packaging_item_ids: Optional[List[str]] = Field(None, description="List of packaging item IDs from packaging_items table")


@app.options("/meal-emissions")
async def meal_emissions_options(request: Request):
    """Handle CORS preflight for meal-emissions endpoint."""
    origin = request.headers.get("origin")
    logger.info(f"OPTIONS request received for /meal-emissions from {origin}")
    # CORS middleware should handle headers, just return empty response
    response = Response(status_code=204)
    return response


@app.post("/meal-emissions", response_model=ComputeResponse)
def meal_emissions(req: MealEmissionsRequest):
    """Compute emissions for a meal item using preset components and category factors."""
    preload_factors()

    if not req.item_id and not req.item_name:
        raise HTTPException(status_code=400, detail="Provide item_id or item_name")

    # 1) Load the meal item
    try:
        if req.item_id:
            item_res = sb.table("items").select("id,name,leftover_g,item_type").eq("id", req.item_id).single().execute()
        else:
            item_res = sb.table("items").select("id,name,leftover_g,item_type").eq("name", req.item_name).single().execute()
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Item not found: {e}")

    if not item_res.data:
        raise HTTPException(status_code=404, detail="Item not found")

    item = item_res.data
    if item.get("item_type") != "meal":
        raise HTTPException(status_code=400, detail="Item is not a meal")

    meal_name = item["name"]
    meal_leftover_g = float(item.get("leftover_g") or 0)

    # 2) Load preset components for this meal
    comps_res = sb.table("preset_meal_components").select(
        "component_name,percentage,grams,emission_factor_category"
    ).eq("meal_name", meal_name).execute()

    components = comps_res.data or []
    if not components:
        raise HTTPException(status_code=404, detail=f"No preset components for meal {meal_name}")

    # Fallback share calc by grams if percentage is missing
    total_grams = sum(float(c.get("grams") or 0) for c in components)

    items_out: List[ItemResult] = []
    total_landfill = 0.0
    total_compost = 0.0
    total_food_avoided = 0.0

    for c in components:
        comp_name = c.get("component_name") or "Unknown"
        category = c.get("emission_factor_category") or comp_name

        # Determine share (0-1)
        pct_val = c.get("percentage")
        if pct_val is not None:
            try:
                share01 = float(pct_val) / 100.0
            except Exception:
                share01 = 0.0
        else:
            grams_val = float(c.get("grams") or 0)
            share01 = (grams_val / total_grams) if total_grams > 0 else 0.0

        # Factor lookup by category
        f = get_factor(str(category))
        if not f:
            # If no category factor, treat as zero-impact but still include row
            component_leftover_g = meal_leftover_g * share01
            items_out.append(ItemResult(
                component_name=comp_name,
                category=str(category),
                share=round(share01, 6),
                component_leftover_g=round(component_leftover_g, 6),
                landfill_kgco2e=0.0,
                compost_kgco2e=0.0,
                avoided_kgco2e=0.0,
                landfill_factor_kg_per_kg=0.0,
                compost_factor_kg_per_kg=0.0,
            ))
            continue

        landfill_factor = mtco2e_per_ton_to_kg_per_kg(f["landfill_mtco2e_ton"]) if f.get("landfill_mtco2e_ton") is not None else 0.0
        compost_factor  = mtco2e_per_ton_to_kg_per_kg(f["compost_mtco2e_ton"]) if f.get("compost_mtco2e_ton") is not None else 0.0
        leftover_pct    = float(f.get("leftover_percentage") or 1.0)

        # Component leftover adjusted by category-specific leftover percentage
        component_leftover_g = meal_leftover_g * share01 * leftover_pct
        kg = component_leftover_g / 1000.0

        landfill = kg * landfill_factor
        compost = kg * compost_factor
        avoided = landfill - compost

        items_out.append(ItemResult(
            component_name=comp_name,
            category=str(category),
            share=round(share01, 6),
            component_leftover_g=round(component_leftover_g, 6),
            landfill_kgco2e=round(landfill, 6),
            compost_kgco2e=round(compost, 6),
            avoided_kgco2e=round(avoided, 6),
            landfill_factor_kg_per_kg=round(landfill_factor, 6),
            compost_factor_kg_per_kg=round(compost_factor, 6),
        ))

        total_landfill += landfill
        total_compost += compost
        total_food_avoided += avoided

    # Process packaging items
    packaging_items = load_packaging_items(req.packaging_item_ids)
    total_packaging_g = 0.0
    total_packaging_avoided = 0.0
    for pkg in packaging_items:
        pkg_name = pkg.get("name") or "Unknown Packaging"
        pkg_grams = float(pkg.get("grams") or 0)
        pkg_category = pkg.get("emission_factor_category") or "Mixed Paper (general)"
        
        total_packaging_g += pkg_grams
        
        # Get emission factor for packaging category
        f = get_factor(str(pkg_category))
        if not f:
            # If no factor found, treat as zero impact but still include row
            items_out.append(ItemResult(
                component_name=pkg_name,
                category=str(pkg_category),
                share=1.0,
                component_leftover_g=pkg_grams,
                landfill_kgco2e=0.0,
                compost_kgco2e=0.0,  # Using compost_kgco2e field to store recycle_kgco2e for packaging
                avoided_kgco2e=0.0,
                landfill_factor_kg_per_kg=0.0,
                compost_factor_kg_per_kg=0.0,  # Storing recycle factor here for packaging
            ))
            continue
        
        # For packaging, use landfill_mtco2e_ton and recycle_mtco2e_ton
        landfill_factor = mtco2e_per_ton_to_kg_per_kg(f["landfill_mtco2e_ton"]) if f.get("landfill_mtco2e_ton") is not None else 0.0
        recycle_factor = mtco2e_per_ton_to_kg_per_kg(f["recycle_mtco2e_ton"]) if f.get("recycle_mtco2e_ton") is not None else 0.0
        
        kg = pkg_grams / 1000.0
        landfill = kg * landfill_factor
        recycle = kg * recycle_factor  # Stored in compost_kgco2e field
        avoided = landfill - recycle
        
        items_out.append(ItemResult(
            component_name=pkg_name,
            category=str(pkg_category),
            share=1.0,
            component_leftover_g=pkg_grams,
            landfill_kgco2e=round(landfill, 6),
            compost_kgco2e=round(recycle, 6),  # Storing recycle emissions here for packaging
            avoided_kgco2e=round(avoided, 6),
            landfill_factor_kg_per_kg=round(landfill_factor, 6),
            compost_factor_kg_per_kg=round(recycle_factor, 6),  # Storing recycle factor here
        ))
        
        total_landfill += landfill
        total_compost += recycle  # Add recycle emissions to compost total (representing recycling)
        total_packaging_avoided += avoided

    # Add packaging grams to meal leftover grams
    meal_leftover_g_with_packaging = meal_leftover_g + total_packaging_g

    totals = Totals(
        landfill_kgco2e=round(total_landfill, 6),
        compost_kgco2e=round(total_compost, 6),
        avoided_kgco2e=round(total_food_avoided + total_packaging_avoided, 6),
        avoided_kgco2e_food=round(total_food_avoided, 6),
        avoided_kgco2e_packaging=round(total_packaging_avoided, 6),
    )

    db_error = None
    db_row_id = None
    try:
        insert_data = {
            "user_id": req.userId,
            "session_id": req.session_id,
            "meal_name": [meal_name] if meal_name else None,
            "leftover_g": meal_leftover_g_with_packaging,
            "items": [i.model_dump() for i in items_out],
            "totals": totals.model_dump(),
            "source": "meal_emissions",
            "warm_version": "v16",
        }
        logger.info(f"Inserting into emissions_results: {insert_data}")
        db_row = sb.table("emissions_results").insert(insert_data).execute()
        logger.info(f"Database response: {db_row}")
        if db_row.data and len(db_row.data) > 0:
            db_row_id = db_row.data[0].get("id")
            logger.info(f"Successfully inserted row with id: {db_row_id}")
        else:
            logger.warning("Database insert returned no data")
            db_error = "Insert returned no data"
    except Exception as e:
        db_error = str(e)
        logger.error(f"Database insert error: {e}", exc_info=True)

    return ComputeResponse(
        session_id=req.session_id,
        meal_name=[meal_name] if meal_name else None,
        leftover_g=meal_leftover_g_with_packaging,
        items=items_out,
        totals=totals,
        db_row_id=db_row_id,
        db_error=db_error,
    )

# ----- Core endpoint -----
@app.options("/calculate-emissions")
async def calculate_emissions_options(request: Request):
    """Handle CORS preflight for calculate-emissions endpoint."""
    origin = request.headers.get("origin")
    logger.info(f"OPTIONS request received for /calculate-emissions from {origin}")
    # CORS middleware should handle headers, just return empty response
    response = Response(status_code=204)
    return response


@app.post("/calculate-emissions", response_model=ComputeResponse)
def calculate_emissions(data: Payload):
    preload_factors()  # ensure cache is ready

    items_out: List[ItemResult] = []
    total_landfill = 0.0
    total_compost = 0.0
    total_food_avoided = 0.0

    total_leftover_g = 0.0
    for comp in data.components:
        f = get_factor(comp.name)
        if not f:
            # no factor found anywhere; record a zero row
            items_out.append(ItemResult(
                component_name=comp.name,
                category=comp.name,
                share=1.0,  # single component gets full share
                component_leftover_g=comp.leftover_g,
                landfill_kgco2e=0.0,
                compost_kgco2e=0.0,
                avoided_kgco2e=0.0,
                landfill_factor_kg_per_kg=0.0,
                compost_factor_kg_per_kg=0.0,
            ))
            total_leftover_g += comp.leftover_g
            continue

        landfill_factor = mtco2e_per_ton_to_kg_per_kg(f["landfill_mtco2e_ton"])
        compost_factor  = mtco2e_per_ton_to_kg_per_kg(f["compost_mtco2e_ton"])

        kg = comp.leftover_g / 1000.0
        landfill = kg * landfill_factor
        compost  = kg * compost_factor
        avoided  = landfill - compost

        items_out.append(ItemResult(
            component_name=comp.name,
            category=comp.name,
            share=1.0,  # single component gets full share
            component_leftover_g=comp.leftover_g,
            landfill_kgco2e=round(landfill, 6),
            compost_kgco2e=round(compost, 6),
            avoided_kgco2e=round(avoided, 6),
            landfill_factor_kg_per_kg=round(landfill_factor, 6),
            compost_factor_kg_per_kg=round(compost_factor, 6),
        ))
        
        total_leftover_g += comp.leftover_g

        total_landfill += landfill
        total_compost  += compost
        total_food_avoided += avoided

    # Process packaging items
    packaging_items = load_packaging_items(data.packaging_item_ids)
    total_packaging_g = 0.0
    total_packaging_avoided = 0.0
    for pkg in packaging_items:
        pkg_name = pkg.get("name") or "Unknown Packaging"
        pkg_grams = float(pkg.get("grams") or 0)
        pkg_category = pkg.get("emission_factor_category") or "Mixed Paper (general)"
        
        total_packaging_g += pkg_grams
        
        # Get emission factor for packaging category
        f = get_factor(str(pkg_category))
        if not f:
            # If no factor found, treat as zero impact but still include row
            items_out.append(ItemResult(
                component_name=pkg_name,
                category=str(pkg_category),
                share=1.0,
                component_leftover_g=pkg_grams,
                landfill_kgco2e=0.0,
                compost_kgco2e=0.0,  # Using compost_kgco2e field to store recycle_kgco2e for packaging
                avoided_kgco2e=0.0,
                landfill_factor_kg_per_kg=0.0,
                compost_factor_kg_per_kg=0.0,  # Storing recycle factor here for packaging
            ))
            continue
        
        # For packaging, use landfill_mtco2e_ton and recycle_mtco2e_ton
        landfill_factor = mtco2e_per_ton_to_kg_per_kg(f["landfill_mtco2e_ton"]) if f.get("landfill_mtco2e_ton") is not None else 0.0
        recycle_factor = mtco2e_per_ton_to_kg_per_kg(f["recycle_mtco2e_ton"]) if f.get("recycle_mtco2e_ton") is not None else 0.0
        
        kg = pkg_grams / 1000.0
        landfill = kg * landfill_factor
        recycle = kg * recycle_factor  # Stored in compost_kgco2e field
        avoided = landfill - recycle
        
        items_out.append(ItemResult(
            component_name=pkg_name,
            category=str(pkg_category),
            share=1.0,
            component_leftover_g=pkg_grams,
            landfill_kgco2e=round(landfill, 6),
            compost_kgco2e=round(recycle, 6),  # Storing recycle emissions here for packaging
            avoided_kgco2e=round(avoided, 6),
            landfill_factor_kg_per_kg=round(landfill_factor, 6),
            compost_factor_kg_per_kg=round(recycle_factor, 6),  # Storing recycle factor here
        ))
        
        total_landfill += landfill
        total_compost += recycle  # Add recycle emissions to compost total (representing recycling)
        total_packaging_avoided += avoided

    # Add packaging grams to total leftover grams
    total_leftover_g += total_packaging_g

    totals = Totals(
        landfill_kgco2e=round(total_landfill, 6),
        compost_kgco2e=round(total_compost, 6),
        avoided_kgco2e=round(total_food_avoided + total_packaging_avoided, 6),
        avoided_kgco2e_food=round(total_food_avoided, 6),
        avoided_kgco2e_packaging=round(total_packaging_avoided, 6),
    )

    db_error = None
    db_row_id = None
    try:
        insert_data = {
            "user_id": data.userId,
            "session_id": data.session_id,
            "meal_name": data.selected_items if data.selected_items else None,
            "leftover_g": total_leftover_g,
            "items": [i.model_dump() for i in items_out],
            "totals": totals.model_dump(),
            "source": "lovable",
            "warm_version": "v16"
        }
        logger.info(f"Inserting into emissions_results: {insert_data}")
        db_row = sb.table("emissions_results").insert(insert_data).execute()
        logger.info(f"Database response: {db_row}")
        if db_row.data and len(db_row.data) > 0:
            db_row_id = db_row.data[0].get("id")
            logger.info(f"Successfully inserted row with id: {db_row_id}")
        else:
            logger.warning("Database insert returned no data")
            db_error = "Insert returned no data"
    except Exception as e:
        # don't fail the request; return results and an error string
        db_error = str(e)
        logger.error(f"Database insert error: {e}", exc_info=True)

    return ComputeResponse(
        session_id=data.session_id,
        meal_name=data.selected_items if data.selected_items else None,
        leftover_g=total_leftover_g,
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

@app.get("/test-db")
def test_db():
    """Test endpoint to verify database connection and table structure."""
    try:
        # Try to query the emissions_results table
        result = sb.table("emissions_results").select("*").limit(1).execute()
        return {
            "ok": True,
            "table_exists": True,
            "sample_count": len(result.data) if result.data else 0,
            "columns": list(result.data[0].keys()) if result.data and len(result.data) > 0 else "No rows to inspect"
        }
    except Exception as e:
        logger.error(f"Database test error: {e}", exc_info=True)
        return {
            "ok": False,
            "error": str(e),
            "table_exists": False
        }