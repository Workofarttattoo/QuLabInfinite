from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class MaterialProperty(BaseModel):
    material_id: str = Field(..., description="Unique identifier for the material")
    formula: Optional[str] = Field(None, description="Chemical formula, e.g., H2O")
    property_name: str = Field(..., description="Name of the property, e.g., Cp, Hf, S")
    value: float = Field(..., description="Numeric value of the property")
    units: str = Field(..., description="Units for the value, e.g., J/mol-K")
    temperature_k: Optional[float] = Field(None, description="Temperature in Kelvin for temperature-dependent properties")
    source: str = Field(..., description="Name of the data source")
    source_ref: Optional[str] = Field(None, description="Reference or URL to the data point")


class CanonicalRecord(BaseModel):
    properties: list[MaterialProperty]
