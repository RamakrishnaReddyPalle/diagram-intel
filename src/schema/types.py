# pydantic models: Component, Port, Wire, Net, Sheet
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Tuple

Coord = Tuple[float, float]
BBox  = Tuple[float, float, float, float]

class Port(BaseModel):
    port_id: str
    label: Optional[str] = None
    phase_tag: Optional[Literal["L1","L2","L3","N","PE"]] = None
    side: Optional[Literal["left","right","top","bottom"]] = None
    idx: Optional[int] = None
    bbox: Optional[BBox] = None

class Component(BaseModel):
    id: str
    name: Optional[str]
    type: Optional[str]
    bbox: BBox
    rotation: Optional[float] = 0.0
    attributes: dict = {}
    ports: List[Port] = []

class WireSegment(BaseModel):
    id: str
    polyline: List[Coord]
    label: Optional[str] = None
    attributes: dict = {}
    touches: List[str] = []  # component.port ids or junction ids

class Junction(BaseModel):
    id: str
    bbox: Optional[BBox] = None
    connected: List[str] = []

class Sheet(BaseModel):
    id: str
    size: Tuple[int,int]
    legend: dict = {}

class CandidateAlt(BaseModel):
    type: Optional[str]
    confidence: float = 0.0

class ComponentCandidate(BaseModel):
    id: str                  # e.g., "{pdf}:{page}:meso:rXXXcYYY"
    pdf: str
    page: int
    tile_path: str
    tile_bbox: BBox
    type: Optional[str] = None
    confidence: float = 0.0
    ports_expected: list = []
    notes: Optional[str] = None
    alternatives: List[CandidateAlt] = []
    labels_context: List[str] = []      # labels used in prompt
    source_model: Optional[str] = None
