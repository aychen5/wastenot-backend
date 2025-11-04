# app.py
import os
import re
import logging
from typing import List, Optional, Dict, Literal
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Request, Query
from starlette.requests import Request
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from supabase import create_client, Client
import httpx
import pandas as pd
import numpy as np
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]  # keep server-side only
sb: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# ----- FastAPI app + CORS -----
app = FastAPI(title="WasteNotNYC Emissions API", version="1.1.0")

# Get allowed origins from environment or use defaults
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "").split(",") if os.environ.get("ALLOWED_ORIGINS") else []
# Add default origins if not in env
default_origins = [
    "https://a74f6327-764b-47f2-8e5a-0cf826e093a1.lovable.app",
    "https://a74f6327-764b-47f2-8e5a-0cf826e093a1.lovableproject.com",
    "https://waste-not-nyc-app.com",  # live production site
    "http://localhost:3000",  # for local testing
    "http://localhost:5173",  # common Vite dev server port
    "http://localhost:8080",  # additional common dev port
]

# Combine env origins with defaults, removing empty strings
all_origins = [origin for origin in ALLOWED_ORIGINS + default_origins if origin.strip()]

# Helper function to check if origin is allowed (must be defined before middleware)
def is_allowed_origin(origin: Optional[str]) -> bool:
    """Check if origin is in allowed list."""
    if not origin:
        return False
    
    # Get allowed origins from environment or use defaults
    env_origins = os.environ.get("ALLOWED_ORIGINS", "").split(",") if os.environ.get("ALLOWED_ORIGINS") else []
    default_origins = [
        "https://a74f6327-764b-47f2-8e5a-0cf826e093a1.lovable.app",
        "https://a74f6327-764b-47f2-8e5a-0cf826e093a1.lovableproject.com",
        "https://waste-not-nyc-app.com",  # live production site
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
    ]
    
    allowed_origins = [o.strip() for o in env_origins + default_origins if o.strip()]
    
    if origin in allowed_origins:
        return True
    # Check regex pattern for lovable.app and lovableproject.com subdomains
    lovable_pattern = r"https://.*\.lovable\.app"
    lovableproject_pattern = r"https://.*\.lovableproject\.com"
    if re.match(lovable_pattern, origin) or re.match(lovableproject_pattern, origin):
        return True
    # Allow waste-not-nyc-app.com domain (with or without www)
    if origin.startswith("https://waste-not-nyc-app.com") or origin.startswith("https://www.waste-not-nyc-app.com"):
        return True
    # For development, allow localhost with any port
    if origin.startswith("http://localhost:") or origin.startswith("https://localhost:"):
        return True
    return False

# Add CORS middleware with permissive settings
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://.*\.(lovable\.app|lovableproject\.com)",  # Allow all lovable subdomains
    allow_origins=all_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# Add middleware to ensure CORS headers are always added to all responses (runs after CORSMiddleware)
class CORSEnforcerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        origin = request.headers.get("origin")
        
        # Check if origin is allowed
        cors_origin = None
        if origin:
            if is_allowed_origin(origin):
                cors_origin = origin
            else:
                # Log for debugging
                logger.warning(f"Origin not in allowed list: {origin}")
        
        response = await call_next(request)
        
        # Always add CORS headers to all responses if origin is allowed
        # This ensures headers are present even if CORSMiddleware doesn't add them
        if cors_origin:
            # Override any existing headers to ensure they're set
            response.headers["Access-Control-Allow-Origin"] = cors_origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, HEAD"
            response.headers["Access-Control-Allow-Headers"] = "*"
            response.headers["Access-Control-Expose-Headers"] = "*"
        elif origin:
            # Log if origin is present but not allowed (for debugging)
            logger.debug(f"Origin present but not allowed: {origin}")
        
        return response

# Add CORS enforcer middleware AFTER CORSMiddleware (so it runs last on responses)
app.add_middleware(CORSEnforcerMiddleware)

# Exception handlers to ensure CORS headers are added to all errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with CORS headers."""
    origin = request.headers.get("origin")
    cors_origin = origin if origin and is_allowed_origin(origin) else None
    headers = {}
    if cors_origin:
        headers = {
            "Access-Control-Allow-Origin": cors_origin,
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, HEAD",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Expose-Headers": "*",
        }
    logger.info(f"Validation error - Origin: {origin}, CORS origin: {cors_origin}, Headers: {headers}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body},
        headers=headers
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with CORS headers."""
    origin = request.headers.get("origin")
    cors_origin = origin if origin and is_allowed_origin(origin) else None
    headers = {}
    if cors_origin:
        headers = {
            "Access-Control-Allow-Origin": cors_origin,
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, HEAD",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Expose-Headers": "*",
        }
    logger.info(f"HTTP exception ({exc.status_code}) - Origin: {origin}, CORS origin: {cors_origin}, Headers: {headers}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=headers
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions with CORS headers."""
    origin = request.headers.get("origin")
    cors_origin = origin if origin and is_allowed_origin(origin) else None
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    headers = {}
    if cors_origin:
        headers = {
            "Access-Control-Allow-Origin": cors_origin,
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, HEAD",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Expose-Headers": "*",
        }
    logger.info(f"General exception - Origin: {origin}, CORS origin: {cors_origin}, Headers: {headers}")
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
    compost_kgco2e: float  # For food: compost emissions; for packaging: recycle emissions
    avoided_kgco2e: float
    landfill_factor_kg_per_kg: float
    compost_factor_kg_per_kg: float  # For food: compost factor; for packaging: recycle factor
    recycle_factor_kg_per_kg: Optional[float] = None  # Only used for packaging items

class Totals(BaseModel):
    landfill_kgco2e: float
    compost_kgco2e: float
    avoided_kgco2e: float
    avoided_kgco2e_food: float
    avoided_kgco2e_packaging: float
    smartphone_charges_equiv: float  # Equivalent number of smartphone charges
    gasoline_car_miles_equiv: float  # Equivalent miles driven by gasoline car

class ComputeResponse(BaseModel):
    session_id: Optional[str] = None
    meal_name: Optional[List[str]] = None
    leftover_g: float
    items: List[ItemResult]
    totals: Totals
    db_row_id: Optional[str] = None
    db_error: Optional[str] = None

# ----- Neighborhood Diversion Rates Schemas -----
class DistrictDiversionData(BaseModel):
    district_id: str
    district_name: str
    boro_cd: Optional[str] = None  # Full BoroCD format (e.g., "306" for Brooklyn 6) for matching with GeoJSON
    diversion_rate: float  # percentage (0-100)
    total_tonnage: float  # total waste collected in tons
    recycled_tonnage: float  # recycled waste (MGP + Paper) in tons
    composted_tonnage: float  # composted/organic waste in tons
    compost_rate: float  # compost rate as percentage (0-100)
    month_year: Optional[str] = None  # most recent month data
    projection_6mo: Optional[float] = None  # projected diversion rate 6 months out
    projection_12mo: Optional[float] = None  # projected diversion rate 12 months out
    latitude: Optional[float] = None  # for Mapbox visualization
    longitude: Optional[float] = None  # for Mapbox visualization

class NeighborhoodDiversionResponse(BaseModel):
    district_type: str
    districts: List[DistrictDiversionData]
    data_source: str
    last_updated: Optional[str] = None
    metadata: Optional[Dict] = None

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

def calculate_energy_equivalencies(avoided_kgco2e: float) -> tuple[float, float]:
    """
    Calculate energy equivalencies for avoided CO2e emissions.
    
    For smartphone charges:
    - Average smartphone charge: 12Wh = 0.012 kWh
    - NY marginal factor (Zone J): 0.54 kg CO₂e/kWh
    - CO2e per charge = 0.012 × 0.54 = 0.00648 kg CO₂e per charge
    
    For gasoline car miles:
    - Average gasoline car: 393 g CO₂e per mile = 0.393 kg CO₂e per mile (EPA, includes CH₄/N₂O)
    
    Returns: (smartphone_charges, gasoline_car_miles)
    """
    # Smartphone calculation
    CO2E_PER_SMARTPHONE_CHARGE_KG = 0.00648  # kg CO2e per charge
    smartphone_charges = avoided_kgco2e / CO2E_PER_SMARTPHONE_CHARGE_KG if avoided_kgco2e > 0 else 0.0
    
    # Gasoline car calculation
    CO2E_PER_GASOLINE_MILE_KG = 0.393  # kg CO2e per mile (393 g per mile from EPA, includes CH₄/N₂O)
    gasoline_car_miles = avoided_kgco2e / CO2E_PER_GASOLINE_MILE_KG if avoided_kgco2e > 0 else 0.0
    
    return smartphone_charges, gasoline_car_miles

def process_packaging_items(
    packaging_items: List[Dict],
    items_out: List[ItemResult],
    total_landfill: float,
    total_compost: float,
    total_packaging_avoided: float
) -> tuple[float, float, float, float]:
    """
    Process packaging items and calculate emissions.
    Returns: (total_packaging_g, updated_total_landfill, updated_total_compost, updated_total_packaging_avoided)
    """
    total_packaging_g = 0.0
    updated_total_landfill = total_landfill
    updated_total_compost = total_compost
    updated_total_packaging_avoided = total_packaging_avoided
    
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
                compost_kgco2e=0.0,  # For packaging: storing recycle emissions
                avoided_kgco2e=0.0,
                landfill_factor_kg_per_kg=0.0,
                compost_factor_kg_per_kg=0.0,
                recycle_factor_kg_per_kg=0.0,
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
            compost_kgco2e=round(recycle, 6),  # For packaging: storing recycle emissions
            avoided_kgco2e=round(avoided, 6),
            landfill_factor_kg_per_kg=round(landfill_factor, 6),
            compost_factor_kg_per_kg=0.0,  # Not used for packaging
            recycle_factor_kg_per_kg=round(recycle_factor, 6),
        ))
        
        updated_total_landfill += landfill
        updated_total_compost += recycle  # Add recycle emissions to compost total (representing recycling)
        updated_total_packaging_avoided += avoided
    
    return total_packaging_g, updated_total_landfill, updated_total_compost, updated_total_packaging_avoided

# ----- NYC OpenData Helpers -----
# NYC DSNY Monthly Tonnage Data - SOCRATA API endpoint
NYC_DSNY_TONNAGE_DATASET_ID = "ebb7-mvp5"  # DSNY Monthly Tonnage Data dataset ID
NYC_SOCRATA_BASE_URL = "https://data.cityofnewyork.us/resource"
# Optional SOCRATA API token - checks multiple environment variable names
# Priority: SOCRATA_APP_TOKEN > SOCRATA_SECRET_TOKEN > NYC_SOCRATA_TOKEN
# Get token from: https://data.cityofnewyork.us/profile/app_tokens
NYC_SOCRATA_TOKEN = (
    os.environ.get("SOCRATA_APP_TOKEN") or 
    os.environ.get("SOCRATA_SECRET_TOKEN")
)

_data_cache: Optional[pd.DataFrame] = None
_cache_timestamp: Optional[datetime] = None
CACHE_DURATION_HOURS = 1  # Cache data for 1 hour

# NYC Community Districts Boundaries from ArcGIS Hub
# Source: https://hub.arcgis.com/datasets/DCP::nyc-community-districts
NYC_COMMUNITY_DISTRICTS_ARCGIS_URL = "https://services.arcgis.com/fYRg49etfc5qh5Jv/arcgis/rest/services/nyc_community_districts/FeatureServer/0/query"
_district_coords_cache: Optional[Dict[str, Dict[str, float]]] = None  # {district_id: {lat: float, lng: float}}
_district_boundaries_cache: Optional[Dict] = None  # Full GeoJSON FeatureCollection

def calculate_centroid(geometry: Dict) -> Optional[tuple[float, float]]:
    """
    Calculate centroid (longitude, latitude) from ArcGIS geometry.
    Handles polygons, multipolygons, and other geometry types.
    Returns (longitude, latitude) or None if unable to calculate.
    """
    try:
        if geometry.get("type") == "Polygon":
            rings = geometry.get("coordinates", [])
            if not rings:
                return None
            # Use the first ring (exterior ring) for centroid calculation
            coords = rings[0]
        elif geometry.get("type") == "MultiPolygon":
            polygons = geometry.get("coordinates", [])
            if not polygons:
                return None
            # Use the first polygon's exterior ring
            coords = polygons[0][0] if polygons else []
        else:
            # Try to extract coordinates directly
            coords = geometry.get("coordinates", [])
            if isinstance(coords[0], (list, tuple)) and len(coords[0]) > 0:
                if isinstance(coords[0][0], (list, tuple)):
                    # Nested coordinates - use first ring
                    coords = coords[0]
        
        if not coords:
            return None
        
        # Calculate centroid as average of all coordinates
        lng_sum = 0.0
        lat_sum = 0.0
        count = 0
        
        for coord in coords:
            if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                lng_sum += float(coord[0])
                lat_sum += float(coord[1])
                count += 1
        
        if count == 0:
            return None
        
        return (lng_sum / count, lat_sum / count)
    
    except Exception as e:
        logger.warning(f"Error calculating centroid: {e}")
        return None

async def fetch_district_coordinates() -> Dict[str, Dict[str, float]]:
    """
    Fetch community district boundaries from ArcGIS Hub and calculate centroids.
    Returns a dictionary mapping district_id to {latitude: float, longitude: float}.
    
    District IDs are formatted as "101", "102", etc. (first digit = borough, next 2 = district).
    Also stores coordinates under district number only (e.g., "1", "2") for flexible matching.
    """
    global _district_coords_cache
    
    # Return cached data if available (coordinates don't change often)
    if _district_coords_cache is not None:
        return _district_coords_cache
    
    try:
        # Try different ArcGIS REST API endpoints
        # The dataset ID from ArcGIS Hub URL can be used to construct the REST endpoint
        # Common patterns for ArcGIS Hub datasets
        urls_to_try = [
            # ArcGIS Hub API v3 - direct GeoJSON download (most reliable)
            "https://opendata.arcgis.com/api/v3/datasets/nyc-community-districts/downloads/data?format=geojson&spatialRefId=4326",
        ]
        
        # Also try REST API endpoints with simpler queries
        rest_endpoints = [
            "https://services.arcgis.com/fYRg49etfc5qh5Jv/arcgis/rest/services/nyc_community_districts/FeatureServer/0/query",
            "https://services.arcgis.com/fYRg49etfc5qh5Jv/arcgis/rest/services/Community_Districts/FeatureServer/0/query",
        ]
        
        district_coords = {}
        
        for url in urls_to_try:
            try:
                logger.info(f"Attempting to fetch district boundaries from: {url}")
                
                async with httpx.AsyncClient(timeout=30.0) as client:
                    # Direct GeoJSON download (ArcGIS Hub API v3)
                    response = await client.get(url)
                    response.raise_for_status()
                    data = response.json()
                    
                    # Handle GeoJSON format
                    if data.get("type") == "FeatureCollection":
                        features = data.get("features", [])
                        for feature in features:
                            props = feature.get("properties", {})
                            geometry = feature.get("geometry", {})
                            
                            # Try different property names for district ID
                            district_id = (
                                props.get("BoroCD") or
                                props.get("boro_cd") or
                                props.get("Boro_CD") or
                                props.get("cd") or
                                props.get("CD") or
                                props.get("BoroCD1") or
                                None
                            )
                            
                            if district_id is None:
                                continue
                            
                            # Convert to string and normalize format
                            district_id_str = str(district_id).strip()
                            
                            # Calculate centroid
                            centroid = calculate_centroid(geometry)
                            if centroid:
                                lng, lat = centroid
                                coord_data = {
                                    "latitude": lat,
                                    "longitude": lng
                                }
                                # Store with full BoroCD format (e.g., "101")
                                district_coords[district_id_str] = coord_data
                                
                                # Also store with district number only (e.g., "1") for flexible matching
                                # Extract district number from BoroCD (last 2 digits)
                                if len(district_id_str) >= 3:
                                    district_num = district_id_str[-2:] if len(district_id_str) > 2 else district_id_str[-1]
                                    # Remove leading zero if present
                                    district_num = str(int(district_num)) if district_num.isdigit() else district_num
                                    district_coords[district_num] = coord_data
                    
                    # Handle ArcGIS REST API response format
                    elif "features" in data:
                        features = data.get("features", [])
                        for feature in features:
                            attrs = feature.get("attributes", {})
                            geometry = feature.get("geometry", {})
                            
                            district_id = (
                                attrs.get("BoroCD") or
                                attrs.get("boro_cd") or
                                attrs.get("Boro_CD") or
                                attrs.get("cd") or
                                attrs.get("CD") or
                                None
                            )
                            
                            if district_id is None:
                                continue
                            
                            district_id_str = str(district_id).strip()
                            
                            # Convert ArcGIS geometry to GeoJSON-like format for centroid calculation
                            if geometry.get("rings"):
                                # Polygon geometry - ArcGIS rings are arrays of [x, y] pairs
                                # For WGS84 (SRID 4326), x = longitude, y = latitude
                                # Convert rings to GeoJSON polygon format
                                rings = geometry.get("rings", [])
                                if rings and len(rings) > 0:
                                    # Use the first ring (exterior ring) for centroid
                                    exterior_ring = rings[0]
                                    geo_json_like = {
                                        "type": "Polygon",
                                        "coordinates": [exterior_ring]  # GeoJSON format expects array of rings
                                    }
                                    centroid = calculate_centroid(geo_json_like)
                                else:
                                    centroid = None
                            elif geometry.get("x") is not None and geometry.get("y") is not None:
                                # Point geometry - use directly
                                # ArcGIS REST API returns x=longitude, y=latitude for WGS84
                                centroid = (float(geometry["x"]), float(geometry["y"]))
                            else:
                                centroid = None
                            
                            if centroid:
                                lng, lat = centroid
                                coord_data = {
                                    "latitude": lat,
                                    "longitude": lng
                                }
                                # Store with full BoroCD format (e.g., "101")
                                district_coords[district_id_str] = coord_data
                                
                                # Also store with district number only (e.g., "1") for flexible matching
                                # Extract district number from BoroCD (last 2 digits)
                                if len(district_id_str) >= 3:
                                    district_num = district_id_str[-2:] if len(district_id_str) > 2 else district_id_str[-1]
                                    # Remove leading zero if present
                                    district_num = str(int(district_num)) if district_num.isdigit() else district_num
                                    district_coords[district_num] = coord_data
                    
                    if district_coords:
                        logger.info(f"Successfully fetched {len(district_coords)} district coordinates from {url}")
                        _district_coords_cache = district_coords
                        return district_coords
                    else:
                        logger.warning(f"No district coordinates found from {url}")
                        continue
                        
            except Exception as e:
                logger.warning(f"Error fetching from {url}: {e}")
                continue
        
        # If ArcGIS Hub API failed, try REST API endpoints with simpler queries
        logger.info("ArcGIS Hub API failed, trying REST API endpoints")
        for rest_url in rest_endpoints:
            try:
                logger.info(f"Attempting REST API endpoint: {rest_url}")
                async with httpx.AsyncClient(timeout=30.0) as client:
                    # Try minimal parameters first
                    params = {
                        "where": "1=1",
                        "f": "json",  # Use JSON format for REST API
                        "outSR": 4326,
                    }
                    response = await client.get(rest_url, params=params)
                    response.raise_for_status()
                    data = response.json()
                    
                    # Process ArcGIS REST API JSON format
                    if "features" in data:
                        features = data.get("features", [])
                        logger.info(f"REST API returned {len(features)} features")
                        for feature in features:
                            attrs = feature.get("attributes", {})
                            geometry = feature.get("geometry", {})
                            
                            district_id = (
                                attrs.get("BoroCD") or
                                attrs.get("boro_cd") or
                                attrs.get("Boro_CD") or
                                attrs.get("cd") or
                                attrs.get("CD") or
                                None
                            )
                            
                            if district_id is None:
                                continue
                            
                            district_id_str = str(district_id).strip()
                            
                            # Convert ArcGIS geometry to GeoJSON-like format for centroid calculation
                            if geometry.get("rings"):
                                rings = geometry.get("rings", [])
                                if rings and len(rings) > 0:
                                    exterior_ring = rings[0]
                                    geo_json_like = {
                                        "type": "Polygon",
                                        "coordinates": [exterior_ring]
                                    }
                                    centroid = calculate_centroid(geo_json_like)
                                else:
                                    centroid = None
                            elif geometry.get("x") is not None and geometry.get("y") is not None:
                                centroid = (float(geometry["x"]), float(geometry["y"]))
                            else:
                                centroid = None
                            
                            if centroid:
                                lng, lat = centroid
                                coord_data = {
                                    "latitude": lat,
                                    "longitude": lng
                                }
                                district_coords[district_id_str] = coord_data
                                
                                # Also store with district number only
                                if len(district_id_str) >= 3:
                                    district_num = district_id_str[-2:] if len(district_id_str) > 2 else district_id_str[-1]
                                    district_num = str(int(district_num)) if district_num.isdigit() else district_num
                                    district_coords[district_num] = coord_data
                        
                        if district_coords:
                            logger.info(f"Successfully fetched {len(district_coords)} district coordinates from REST API")
                            _district_coords_cache = district_coords
                            return district_coords
                        
            except Exception as e:
                logger.warning(f"Error fetching from REST API {rest_url}: {e}")
                continue
        
        # If all URLs failed, return empty dict
        logger.warning("Unable to fetch district coordinates from any source")
        _district_coords_cache = {}
        return {}
        
    except Exception as e:
        logger.error(f"Error fetching district coordinates: {e}", exc_info=True)
        _district_coords_cache = {}
        return {}

async def fetch_nyc_tonnage_data() -> pd.DataFrame:
    """
    Fetch NYC DSNY monthly tonnage data from SOCRATA API.
    Returns a pandas DataFrame with the data.
    
    Uses NYC_SOCRATA_TOKEN if available for higher rate limits (1000 req/hr vs lower without token).
    """
    global _data_cache, _cache_timestamp
    
    # Check cache
    if _data_cache is not None and _cache_timestamp is not None:
        if datetime.now() - _cache_timestamp < timedelta(hours=CACHE_DURATION_HOURS):
            logger.info("Returning cached NYC tonnage data")
            return _data_cache
    
    try:
        # Build SOCRATA API URL with query parameters
        url = f"{NYC_SOCRATA_BASE_URL}/{NYC_DSNY_TONNAGE_DATASET_ID}.json"
        
        # Request limit - get last 2 years of data
        params = {
            "$limit": 50000,  # Adjust based on data volume
            "$order": "month DESC",  # Get most recent first
        }
        
        # Add token if available (for higher rate limits: 1000 req/hr vs lower without token)
        headers = {}
        if NYC_SOCRATA_TOKEN:
            headers["X-App-Token"] = NYC_SOCRATA_TOKEN
            logger.info("Using SOCRATA API token for enhanced rate limits")
        else:
            logger.info("No SOCRATA token found - using public API (lower rate limits)")
        
        logger.info(f"Fetching NYC tonnage data from {url}")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
        
        if not data:
            logger.warning("No data returned from NYC OpenData API")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Cache the data
        _data_cache = df
        _cache_timestamp = datetime.now()
        logger.info(f"Fetched {len(df)} records from NYC OpenData")
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching NYC tonnage data: {e}", exc_info=True)
        # Return cached data if available, otherwise raise
        if _data_cache is not None:
            logger.warning("Using stale cached data due to fetch error")
            return _data_cache
        raise HTTPException(status_code=503, detail=f"Unable to fetch NYC data: {str(e)}")

def calculate_diversion_rates(
    df: pd.DataFrame, 
    district_coords: Optional[Dict[str, Dict[str, float]]] = None
) -> List[DistrictDiversionData]:
    """
    Calculate diversion rates from NYC tonnage data for community districts.
    
    Diversion Rate = (Recycled tons + Composted tons) / Total waste tons × 100
    Where:
    - Recycled = MGPTONSCOLLECTED (metal/glass/plastic) + PAPERTONSCOLLECTED (paper)
    - Composted = RESORGANICSTONS + SCHOOLORGANICTONS + LEAVESORGANICSTONS + OTHERORGANICSTONS (all organic sources)
    - Total = Recycled + Composted + REFUSETONSCOLLECTED (trash)
    
    Args:
        df: DataFrame with NYC tonnage data
        district_coords: Optional dictionary mapping district_id to {latitude: float, longitude: float}
    """
    if df.empty:
        return []
    
    # Find the community district column (case-insensitive search)
    district_col = None
    possible_cols = ["COMMUNITYDISTRICT", "cd", "community_district"]
    df_columns_upper = [c.upper() for c in df.columns]
    
    for col in possible_cols:
        # Try exact match first
        if col in df.columns:
            district_col = col
            break
        # Try case-insensitive match
        if col.upper() in df_columns_upper:
            district_col = df.columns[df_columns_upper.index(col.upper())]
            break
    
    if district_col is None:
        # Try to find any community district-related column (case-insensitive)
        district_cols = [c for c in df.columns if 'COMMUNITYDISTRICT' in c.upper() or 'CD' in c.upper()]
        if district_cols:
            district_col = district_cols[0]
        else:
            logger.warning("No community district column found")
            return []
    
    # Find columns (case-insensitive) - use specific column names from NYC dataset
    df_columns_dict = {c.upper(): c for c in df.columns}
    mgp_col = df_columns_dict.get("MGPTONSCOLLECTED", None)
    paper_col = df_columns_dict.get("PAPERTONSCOLLECTED", None)
    organic_col = df_columns_dict.get("RESORGANICSTONS", None)
    school_organic_col = df_columns_dict.get("SCHOOLORGANICTONS", None)
    leaves_organic_col = df_columns_dict.get("LEAVESORGANICSTONS", None)
    other_organic_col = df_columns_dict.get("OTHERORGANICSTONS", None)
    refuse_col = df_columns_dict.get("REFUSETONSCOLLECTED", None)
    
    if not mgp_col or not paper_col or not organic_col or not refuse_col:
        missing = [col for col, val in [
            ("MGPTONSCOLLECTED", mgp_col),
            ("PAPERTONSCOLLECTED", paper_col),
            ("RESORGANICSTONS", organic_col),
            ("REFUSETONSCOLLECTED", refuse_col)
        ] if not val]
        logger.error(f"Missing required columns: {missing}. Available columns: {list(df.columns)}")
        return []
    
    # Optional additional organic columns - log if they exist
    organic_cols_list = [organic_col]
    if school_organic_col:
        organic_cols_list.append(school_organic_col)
        logger.info(f"Found SCHOOLORGANICTONS column: {school_organic_col}")
    if leaves_organic_col:
        organic_cols_list.append(leaves_organic_col)
        logger.info(f"Found LEAVESORGANICSTONS column: {leaves_organic_col}")
    if other_organic_col:
        organic_cols_list.append(other_organic_col)
        logger.info(f"Found OTHERORGANICSTONS column: {other_organic_col}")
    
    # Convert tonnage columns to numeric
    columns_to_convert = [mgp_col, paper_col, organic_col, refuse_col]
    # Add optional organic columns if they exist
    if school_organic_col:
        columns_to_convert.append(school_organic_col)
    if leaves_organic_col:
        columns_to_convert.append(leaves_organic_col)
    if other_organic_col:
        columns_to_convert.append(other_organic_col)
    
    for col in columns_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Convert district column to numeric
    if district_col in df.columns:
        df[district_col] = pd.to_numeric(df[district_col], errors='coerce').fillna(0)
    
    # Group by district and calculate totals
    district_data = []
    
    grouped = df.groupby(district_col)
    
    for group_key, group_df in grouped:
        district_id_str = str(group_key).strip()
        
        # Calculate tonnages
        mgp = group_df[mgp_col].sum()
        paper = group_df[paper_col].sum()
        organic = group_df[organic_col].sum()
        refuse = group_df[refuse_col].sum()
        
        # Calculate composted tonnage from all organic sources
        composted_tonnage = organic  # RESORGANICSTONS
        # Add additional organic sources if available
        if school_organic_col and school_organic_col in group_df.columns:
            composted_tonnage += group_df[school_organic_col].sum()
        if leaves_organic_col and leaves_organic_col in group_df.columns:
            composted_tonnage += group_df[leaves_organic_col].sum()
        if other_organic_col and other_organic_col in group_df.columns:
            composted_tonnage += group_df[other_organic_col].sum()
        
        # Separated tonnages
        recycled_tonnage = mgp + paper  # Recycled (MGP + Paper)
        diverted_tonnage = recycled_tonnage + composted_tonnage  # Total diverted
        
        # Total waste = diverted + refuse
        total_tonnage = diverted_tonnage + refuse
        
        # Calculate diversion rate: (Recycled + Composted) / Total × 100
        if total_tonnage > 0:
            diversion_rate = (diverted_tonnage / total_tonnage) * 100
            compost_rate = (composted_tonnage / total_tonnage) * 100
        else:
            diversion_rate = 0.0
            compost_rate = 0.0
        
        # Get most recent month
        month_cols = [c for c in df.columns if 'MONTH' in c.upper() or 'MONTH' in c]
        if month_cols:
            try:
                group_df_sorted = group_df.sort_values(month_cols[0], ascending=False)
                recent_month = group_df_sorted.iloc[0][month_cols[0]] if len(group_df_sorted) > 0 else None
            except:
                recent_month = None
        else:
            recent_month = None
        
        # Calculate projections using simple linear trend
        projections = calculate_projections(
            group_df, mgp_col, paper_col, organic_col, refuse_col,
            school_organic_col, leaves_organic_col, other_organic_col
        )
        
        # Get coordinates for this district if available
        # Try multiple matching strategies:
        # 1. Exact match (e.g., "1" -> "1")
        # 2. BoroCD format (e.g., "1" -> try "101", "201", "301", "401", "501")
        # 3. Padded format (e.g., "1" -> "01" -> try "101", "201", etc.)
        latitude = None
        longitude = None
        if district_coords:
            # Try exact match first
            coords = district_coords.get(district_id_str)
            
            # If not found and district_id looks like a single number, try BoroCD formats
            if not coords and district_id_str.isdigit():
                # Try with borough prefixes (1=Manhattan, 2=Bronx, 3=Brooklyn, 4=Queens, 5=Staten Island)
                for borough in range(1, 6):
                    # Try with 2-digit district number (e.g., "01", "10")
                    padded_district = district_id_str.zfill(2)
                    boro_cd = f"{borough}{padded_district}"
                    if boro_cd in district_coords:
                        coords = district_coords[boro_cd]
                        break
                    # Also try without padding (e.g., "11" -> "11", "12" -> "12")
                    boro_cd = f"{borough}{district_id_str}"
                    if boro_cd in district_coords:
                        coords = district_coords[boro_cd]
                        break
            
            if coords:
                latitude = coords.get("latitude")
                longitude = coords.get("longitude")
        
        # Calculate possible BoroCD formats for matching with GeoJSON
        # BoroCD format: first digit = borough (1-5), next 2 digits = district number
        # Since we don't know which borough each district belongs to from tonnage data alone,
        # we'll generate all possible BoroCD formats for this district number
        district_num = int(district_id_str) if district_id_str.isdigit() else None
        possible_boro_cds = []
        primary_boro_cd = None
        
        if district_num and 1 <= district_num <= 18:
            # Try all 5 boroughs (1=Manhattan, 2=Bronx, 3=Brooklyn, 4=Queens, 5=Staten Island)
            for borough in range(1, 6):
                # Format as 3-digit BoroCD (e.g., "106" for Manhattan 6, "306" for Brooklyn 6, "315" for Brooklyn 15)
                padded_district = str(district_num).zfill(2)
                boro_cd = f"{borough}{padded_district}"
                possible_boro_cds.append(boro_cd)
            
            # Use the first one as primary (Manhattan) - frontend should try all matches
            primary_boro_cd = possible_boro_cds[0] if possible_boro_cds else None
        
        district_data.append(DistrictDiversionData(
            district_id=district_id_str,
            district_name=f"Community District {district_id_str}",
            boro_cd=primary_boro_cd,  # Primary BoroCD for matching (will try all 5 boroughs if needed)
            diversion_rate=round(diversion_rate, 2),
            total_tonnage=round(total_tonnage, 2),
            recycled_tonnage=round(recycled_tonnage, 2),  # Recycled (MGP + Paper)
            composted_tonnage=round(composted_tonnage, 2),  # Composted (Organic)
            compost_rate=round(compost_rate, 2),  # Compost rate percentage
            month_year=str(recent_month) if recent_month else None,
            projection_6mo=round(projections['6mo'], 2) if projections else None,
            projection_12mo=round(projections['12mo'], 2) if projections else None,
            latitude=latitude,
            longitude=longitude,
        ))
    
    # Sort by district_id
    district_data.sort(key=lambda x: x.district_id)
    
    return district_data

def calculate_projections(group_df: pd.DataFrame, mgp_col: str, paper_col: str, organic_col: str, refuse_col: str,
                          school_organic_col: Optional[str] = None, leaves_organic_col: Optional[str] = None,
                          other_organic_col: Optional[str] = None) -> Optional[Dict[str, float]]:
    """
    Calculate 6-month and 12-month projections using linear trend analysis.
    Returns projected diversion rates.
    
    Diversion Rate = (Recycled + Composted) / Total × 100
    Where Composted includes all organic sources (RESORGANICSTONS + SCHOOLORGANICTONS + LEAVESORGANICSTONS + OTHERORGANICSTONS)
    """
    if mgp_col not in group_df.columns or paper_col not in group_df.columns or \
       organic_col not in group_df.columns or refuse_col not in group_df.columns:
        return None
    
    try:
        # Sort by month if available
        month_cols = [c for c in group_df.columns if 'MONTH' in c.upper() or 'MONTH' in c]
        if month_cols:
            group_df = group_df.sort_values(month_cols[0])
        
        # Calculate monthly diversion rates
        group_df = group_df.copy()
        # Calculate total organic from all available sources
        total_organic = group_df[organic_col]  # RESORGANICSTONS
        if school_organic_col and school_organic_col in group_df.columns:
            total_organic = total_organic + group_df[school_organic_col]
        if leaves_organic_col and leaves_organic_col in group_df.columns:
            total_organic = total_organic + group_df[leaves_organic_col]
        if other_organic_col and other_organic_col in group_df.columns:
            total_organic = total_organic + group_df[other_organic_col]
        
        # Diverted = MGP + Paper + Total Organic (recycled + composted)
        group_df['diverted'] = group_df[mgp_col] + group_df[paper_col] + total_organic
        # Total = Diverted + Refuse
        group_df['total'] = group_df['diverted'] + group_df[refuse_col]
        # Diversion rate = (Diverted / Total) × 100
        group_df['diversion_rate'] = (group_df['diverted'] / group_df['total'] * 100).replace([np.inf, -np.inf], 0)
        
        # Remove NaN values
        group_df = group_df.dropna(subset=['diversion_rate'])
        
        if len(group_df) < 3:
            # Not enough data for projection
            current_rate = group_df['diversion_rate'].mean() if len(group_df) > 0 else 0
            return {'6mo': current_rate, '12mo': current_rate}
        
        # Simple linear trend: use average of last 3 months
        recent_data = group_df.tail(3)['diversion_rate']
        avg_rate = recent_data.mean()
        trend = (recent_data.iloc[-1] - recent_data.iloc[0]) / len(recent_data) if len(recent_data) > 1 else 0
        
        # Project forward (conservative - assume trend continues but diminishes)
        projection_6mo = avg_rate + (trend * 3)  # 3 months average for 6mo
        projection_12mo = avg_rate + (trend * 6)  # 6 months average for 12mo
        
        # Cap at reasonable values (0-100%)
        projection_6mo = max(0, min(100, projection_6mo))
        projection_12mo = max(0, min(100, projection_12mo))
        
        return {'6mo': projection_6mo, '12mo': projection_12mo}
        
    except Exception as e:
        logger.warning(f"Error calculating projections: {e}")
        # Return current average as fallback
        if all(col in group_df.columns for col in [mgp_col, paper_col, organic_col, refuse_col]):
            # Calculate total organic from all available sources
            total_organic = group_df[organic_col].sum()  # RESORGANICSTONS
            if school_organic_col and school_organic_col in group_df.columns:
                total_organic += group_df[school_organic_col].sum()
            if leaves_organic_col and leaves_organic_col in group_df.columns:
                total_organic += group_df[leaves_organic_col].sum()
            if other_organic_col and other_organic_col in group_df.columns:
                total_organic += group_df[other_organic_col].sum()
            
            diverted = group_df[mgp_col].sum() + group_df[paper_col].sum() + total_organic
            total = diverted + group_df[refuse_col].sum()
            current_rate = (diverted / total * 100) if total > 0 else 0
            return {'6mo': current_rate, '12mo': current_rate}
        return None


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
    # Explicitly add CORS headers
    cors_origin = origin if is_allowed_origin(origin) else None
    headers = {}
    if cors_origin:
        headers = {
            "Access-Control-Allow-Origin": cors_origin,
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, HEAD",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "3600",
        }
    response = Response(status_code=204, headers=headers)
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
                recycle_factor_kg_per_kg=None,
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
            recycle_factor_kg_per_kg=None,
        ))

        total_landfill += landfill
        total_compost += compost
        total_food_avoided += avoided

    # Process packaging items
    packaging_items = load_packaging_items(req.packaging_item_ids)
    total_packaging_g, total_landfill, total_compost, total_packaging_avoided = process_packaging_items(
        packaging_items, items_out, total_landfill, total_compost, 0.0
    )

    # Add packaging grams to meal leftover grams
    meal_leftover_g_with_packaging = meal_leftover_g + total_packaging_g

    # Calculate energy equivalencies
    total_avoided = total_food_avoided + total_packaging_avoided
    smartphone_charges, gasoline_car_miles = calculate_energy_equivalencies(total_avoided)

    totals = Totals(
        landfill_kgco2e=round(total_landfill, 6),
        compost_kgco2e=round(total_compost, 6),
        avoided_kgco2e=round(total_avoided, 6),
        avoided_kgco2e_food=round(total_food_avoided, 6),
        avoided_kgco2e_packaging=round(total_packaging_avoided, 6),
        smartphone_charges_equiv=round(smartphone_charges, 2),
        gasoline_car_miles_equiv=round(gasoline_car_miles, 2),
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
    # Explicitly add CORS headers
    cors_origin = origin if origin and is_allowed_origin(origin) else None
    headers = {}
    if cors_origin:
        headers = {
            "Access-Control-Allow-Origin": cors_origin,
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, HEAD",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "3600",
        }
    else:
        logger.warning(f"OPTIONS request from non-allowed origin: {origin}")
    response = Response(status_code=204, headers=headers)
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
                recycle_factor_kg_per_kg=None,
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
            recycle_factor_kg_per_kg=None,
        ))
        
        total_leftover_g += comp.leftover_g

        total_landfill += landfill
        total_compost  += compost
        total_food_avoided += avoided

    # Process packaging items
    packaging_items = load_packaging_items(data.packaging_item_ids)
    total_packaging_g, total_landfill, total_compost, total_packaging_avoided = process_packaging_items(
        packaging_items, items_out, total_landfill, total_compost, 0.0
    )

    # Add packaging grams to total leftover grams
    total_leftover_g += total_packaging_g

    # Calculate energy equivalencies
    total_avoided = total_food_avoided + total_packaging_avoided
    smartphone_charges, gasoline_car_miles = calculate_energy_equivalencies(total_avoided)

    totals = Totals(
        landfill_kgco2e=round(total_landfill, 6),
        compost_kgco2e=round(total_compost, 6),
        avoided_kgco2e=round(total_avoided, 6),
        avoided_kgco2e_food=round(total_food_avoided, 6),
        avoided_kgco2e_packaging=round(total_packaging_avoided, 6),
        smartphone_charges_equiv=round(smartphone_charges, 2),
        gasoline_car_miles_equiv=round(gasoline_car_miles, 2),
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

# ----- Neighborhood Diversion Rates Endpoint -----
@app.options("/neighborhood-diversion-rates")
async def neighborhood_diversion_rates_options(request: Request):
    """Handle CORS preflight for neighborhood-diversion-rates endpoint."""
    origin = request.headers.get("origin")
    logger.info(f"OPTIONS request received for /neighborhood-diversion-rates from {origin}")
    # Explicitly add CORS headers
    cors_origin = origin if is_allowed_origin(origin) else None
    headers = {}
    if cors_origin:
        headers = {
            "Access-Control-Allow-Origin": cors_origin,
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, HEAD",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "3600",
        }
    response = Response(status_code=204, headers=headers)
    return response

@app.get("/neighborhood-diversion-rates", response_model=NeighborhoodDiversionResponse)
async def neighborhood_diversion_rates():
    """
    Calculate diversion rates for NYC community districts using monthly tonnage data.
    
    Diversion rate = (Recycled tons + Composted tons) / Total waste tons × 100
    Where:
    - Recycled = MGPTONSCOLLECTED (metal/glass/plastic) + PAPERTONSCOLLECTED (paper)
    - Composted = RESORGANICSTONS + SCHOOLORGANICTONS + LEAVESORGANICSTONS + OTHERORGANICSTONS (all organic sources)
    - Total = Recycled + Composted + REFUSETONSCOLLECTED (trash)
    
    Returns diversion rates and projections for each community district, suitable for Mapbox visualization.
    """
    try:
        # Always use community districts
        district_type = "community"
        
        # Fetch NYC tonnage data
        logger.info("Fetching diversion rates for community districts")
        df = await fetch_nyc_tonnage_data()
        
        if df.empty:
            logger.warning("No data available from NYC OpenData")
            return NeighborhoodDiversionResponse(
                district_type=district_type,
                districts=[],
                data_source="NYC OpenData (DSNY Monthly Tonnage Data)",
                last_updated=None,
                metadata={"error": "No data available"}
            )
        
        # Fetch district coordinates for community districts (for map visualization)
        district_coords = None
        try:
            district_coords = await fetch_district_coordinates()
            if district_coords:
                logger.info(f"Fetched coordinates for {len(district_coords)} community districts")
            else:
                logger.warning("No district coordinates available - map visualization will not have lat/lng")
        except Exception as e:
            logger.warning(f"Error fetching district coordinates: {e} - continuing without coordinates")
        
        # Calculate diversion rates
        districts = calculate_diversion_rates(df, district_coords)
        
        if not districts:
            logger.warning("No community districts found")
            return NeighborhoodDiversionResponse(
                district_type=district_type,
                districts=[],
                data_source="NYC OpenData (DSNY Monthly Tonnage Data)",
                last_updated=_cache_timestamp.isoformat() if _cache_timestamp else None,
                metadata={"warning": "No districts found", "columns": list(df.columns)}
            )
        
        # Get cache timestamp
        last_updated = _cache_timestamp.isoformat() if _cache_timestamp else datetime.now().isoformat()
        
        # Create BoroCD lookup map: BoroCD -> district_id
        # This helps frontend match GeoJSON features (which use BoroCD) to diversion rate data
        boro_cd_lookup = {}
        for district in districts:
            district_num = int(district.district_id) if district.district_id.isdigit() else None
            if district_num and 1 <= district_num <= 18:
                # Generate all possible BoroCD formats for this district
                for borough in range(1, 6):
                    padded_district = str(district_num).zfill(2)
                    boro_cd = f"{borough}{padded_district}"
                    boro_cd_lookup[boro_cd] = district.district_id
        
        return NeighborhoodDiversionResponse(
            district_type=district_type,
            districts=districts,
            data_source="NYC OpenData (DSNY Monthly Tonnage Data)",
            last_updated=last_updated,
            boro_cd_lookup=boro_cd_lookup,
            metadata={
                "total_districts": len(districts),
                "dataset_id": NYC_DSNY_TONNAGE_DATASET_ID,
                "api_url": f"{NYC_SOCRATA_BASE_URL}/{NYC_DSNY_TONNAGE_DATASET_ID}.json"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating neighborhood diversion rates: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
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

@app.get("/test-district-coordinates")
async def test_district_coordinates():
    """Test endpoint to verify district coordinates are being fetched from ArcGIS."""
    try:
        coords = await fetch_district_coordinates()
        sample_coords = dict(list(coords.items())[:5]) if coords else {}
        return {
            "ok": True,
            "total_districts": len(coords),
            "sample_coordinates": sample_coords,
            "has_coordinates": len(coords) > 0
        }
    except Exception as e:
        logger.error(f"District coordinates test error: {e}", exc_info=True)
        return {
            "ok": False,
            "error": str(e),
            "has_coordinates": False
        }

@app.post("/test-emissions-calculation")
def test_emissions_calculation(data: Payload):
    """Test endpoint to verify emissions calculations are working correctly."""
    try:
        # Call the actual calculation endpoint
        result = calculate_emissions(data)
        
        # Return a detailed breakdown for debugging
        return {
            "ok": True,
            "response": result.model_dump(),
            "totals_breakdown": {
                "landfill_kgco2e": result.totals.landfill_kgco2e,
                "compost_kgco2e": result.totals.compost_kgco2e,
                "avoided_kgco2e": result.totals.avoided_kgco2e,
                "avoided_kgco2e_food": result.totals.avoided_kgco2e_food,
                "avoided_kgco2e_packaging": result.totals.avoided_kgco2e_packaging,
            },
            "items_count": len(result.items),
            "sample_item": result.items[0].model_dump() if result.items else None
        }
    except Exception as e:
        logger.error(f"Test emissions calculation error: {e}", exc_info=True)
        return {
            "ok": False,
            "error": str(e)
        }

@app.options("/community-districts-geojson")
async def community_districts_geojson_options(request: Request):
    """Handle CORS preflight for community-districts-geojson endpoint."""
    origin = request.headers.get("origin")
    logger.info(f"OPTIONS request received for /community-districts-geojson from {origin}")
    # Explicitly add CORS headers
    cors_origin = origin if is_allowed_origin(origin) else None
    headers = {}
    if cors_origin:
        headers = {
            "Access-Control-Allow-Origin": cors_origin,
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, HEAD",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "3600",
        }
    response = Response(status_code=204, headers=headers)
    return response

@app.get("/community-districts-geojson")
async def get_community_districts_geojson():
    """
    Get full GeoJSON boundaries for NYC community districts.
    Returns a GeoJSON FeatureCollection that can be used directly in map visualizations.
    Reads from local file: community_districts.geojson
    """
    global _district_boundaries_cache
    
    # Return cached data if available
    if _district_boundaries_cache is not None:
        return Response(
            content=json.dumps(_district_boundaries_cache),
            media_type="application/geo+json"
        )
    
    # Try multiple possible filenames
    possible_filenames = [
        "NYC_Community_Districts_147499644259232207.geojson",
        "community_districts.geojson",
        "nyc_community_districts.geojson"
    ]
    
    try:
        # Try to load from local file first
        geojson_file_path = None
        for filename in possible_filenames:
            file_path = os.path.join(os.path.dirname(__file__), filename)
            if os.path.exists(file_path):
                geojson_file_path = file_path
                break
        
        if geojson_file_path:
            logger.info(f"Loading GeoJSON from local file: {geojson_file_path}")
            with open(geojson_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate it's a GeoJSON FeatureCollection
            if data.get("type") == "FeatureCollection":
                _district_boundaries_cache = data
                logger.info(f"Successfully loaded GeoJSON with {len(data.get('features', []))} districts from local file")
                return Response(
                    content=json.dumps(data),
                    media_type="application/geo+json"
                )
            else:
                logger.warning(f"Invalid GeoJSON format in local file: {data.get('type')}")
                raise HTTPException(status_code=500, detail="Invalid GeoJSON format in local file")
        else:
            # Fallback: Try to fetch from ArcGIS Hub API if local file doesn't exist
            logger.warning(f"Local GeoJSON file not found. Tried: {possible_filenames}. Attempting to fetch from ArcGIS")
            url = "https://opendata.arcgis.com/api/v3/datasets/nyc-community-districts/downloads/data?format=geojson&spatialRefId=4326"
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
                
                # Validate it's a GeoJSON FeatureCollection
                if data.get("type") == "FeatureCollection":
                    _district_boundaries_cache = data
                    logger.info(f"Successfully fetched GeoJSON with {len(data.get('features', []))} districts from ArcGIS")
                    return Response(
                        content=json.dumps(data),
                        media_type="application/geo+json"
                    )
                else:
                    logger.warning(f"Unexpected response format: {data.get('type')}")
                    raise HTTPException(status_code=500, detail="Invalid GeoJSON format from ArcGIS")
                
    except FileNotFoundError:
        logger.error(f"GeoJSON file not found. Tried: {possible_filenames}")
        raise HTTPException(status_code=404, detail="Community districts GeoJSON file not found. Please add the GeoJSON file to the project root.")
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing GeoJSON file: {e}")
        raise HTTPException(status_code=500, detail=f"Error parsing GeoJSON file: {str(e)}")
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching GeoJSON: {e}")
        raise HTTPException(status_code=503, detail=f"Unable to fetch district boundaries: {str(e)}")
    except Exception as e:
        logger.error(f"Error loading district boundaries: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error loading district boundaries: {str(e)}")