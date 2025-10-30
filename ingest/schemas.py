"""
Pydantic schemas for QuLab data structures.

Defines validation schemas for teleportation results, governance data,
and encoding operations with proper type checking and serialization.
"""

from typing import List, Dict, Optional, Union, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator
import numpy as np
import hashlib
import json


class Provenance(BaseModel):
    source: str
    url: Optional[str] = None
    license: str = "UNKNOWN"
    acquired_at: datetime = Field(default_factory=datetime.utcnow)
    notes: Optional[str] = None
    extra: Dict[str, Any] = {}


class RecordChem(BaseModel):
    # Canonical lab record: PV/T with optional composition/phase
    substance: str
    phase: Optional[str] = None
    pressure_pa: float
    temperature_k: float
    volume_m3_per_mol: Optional[float] = None
    enthalpy_j_per_mol: Optional[float] = None
    entropy_j_per_mol_k: Optional[float] = None
    composition: Optional[Dict[str, float]] = None  # mole fractions
    experiment_id: Optional[str] = None
    tags: List[str] = []
    provenance: Provenance

    def content_hash(self) -> str:
        payload = self.model_dump()
        payload['acquired_at'] = self.provenance.acquired_at.isoformat()
        s = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(s).hexdigest()


class RecordMaterial(BaseModel):
    # Canonical lab record for a solid material
    substance: str
    material_id: str
    phase: str = "solid"
    
    # key properties
    structure: Dict[str, Any]
    formation_energy_per_atom_ev: float
    
    # optional properties
    band_gap_ev: Optional[float] = None
    density_g_cm3: Optional[float] = None
    volume_a3_per_atom: Optional[float] = None

    tags: List[str] = []
    provenance: Provenance

    def content_hash(self) -> str:
        payload = self.model_dump()
        payload['provenance']['acquired_at'] = self.provenance.acquired_at.isoformat()
        # structure can have float values that are not perfectly reproducible
        # in json, so we need to be careful
        s = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(s).hexdigest()
        

class TeleportationSchema(BaseModel):
    """Schema for teleportation experiment results."""
    
    experiment_id: str = Field(..., description="Unique experiment identifier")
    timestamp: datetime = Field(..., description="Experiment timestamp")
    alpha: float = Field(..., ge=0.0, le=1.0, description="Amplitude of |0⟩ state")
    beta: float = Field(..., ge=0.0, le=1.0, description="Amplitude of |1⟩ state")
    fidelity: float = Field(..., ge=0.0, le=1.0, description="Teleportation fidelity")
    success_probability: float = Field(..., ge=0.0, le=1.0, description="Success probability")
    shots: int = Field(..., gt=0, description="Number of measurement shots")
    execution_time: float = Field(..., ge=0.0, description="Execution time in seconds")
    measurement_results: Dict[str, int] = Field(..., description="Bell measurement outcomes")
    classical_bits: List[int] = Field(..., description="Classical bits sent to Bob")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('beta')
    def validate_normalization(cls, v, values):
        """Validate that state is normalized."""
        if 'alpha' in values:
            alpha = values['alpha']
            if not np.isclose(abs(alpha)**2 + abs(v)**2, 1.0, atol=1e-10):
                raise ValueError("State must be normalized: |α|² + |β|² = 1")
        return v
    
    class Config:
        arbitrary_types_allowed = True


class GovernanceSchema(BaseModel):
    """Schema for governance and evidence data."""
    
    ledger_id: str = Field(..., description="Evidence ledger identifier")
    timestamp: datetime = Field(..., description="Evidence timestamp")
    alpha_prior: float = Field(..., gt=0.0, description="Prior alpha parameter")
    beta_prior: float = Field(..., gt=0.0, description="Prior beta parameter")
    alpha_posterior: float = Field(..., gt=0.0, description="Posterior alpha parameter")
    beta_posterior: float = Field(..., gt=0.0, description="Posterior beta parameter")
    mean_fidelity: float = Field(..., ge=0.0, le=1.0, description="Mean fidelity estimate")
    std_fidelity: float = Field(..., ge=0.0, description="Standard deviation of fidelity")
    credible_interval_95: List[float] = Field(..., description="95% credible interval")
    evidence_entries: List[Dict[str, Any]] = Field(..., description="Evidence entries")
    total_shots: int = Field(..., ge=0, description="Total measurement shots")
    total_successes: int = Field(..., ge=0, description="Total successful teleportations")
    
    class Config:
        arbitrary_types_allowed = True


class EncodingSchema(BaseModel):
    """Schema for base-N encoding operations."""
    
    operation_id: str = Field(..., description="Unique operation identifier")
    timestamp: datetime = Field(..., description="Operation timestamp")
    base: int = Field(..., ge=2, le=1024, description="Encoding base")
    alphabet: str = Field(..., description="Encoding alphabet")
    original_data: str = Field(..., description="Original data (base64 encoded)")
    encoded_data: str = Field(..., description="Encoded data")
    original_length: int = Field(..., ge=0, description="Original data length")
    encoded_length: int = Field(..., ge=0, description="Encoded data length")
    compression_ratio: float = Field(..., ge=0.0, description="Compression ratio")
    padding: int = Field(..., ge=0, description="Padding characters")
    error_detection: bool = Field(..., description="Error detection enabled")
    efficiency_metrics: Dict[str, float] = Field(..., description="Efficiency metrics")
    
    class Config:
        arbitrary_types_allowed = True


class SimulationSchema(BaseModel):
    """Schema for quantum simulation results."""
    
    simulation_id: str = Field(..., description="Unique simulation identifier")
    timestamp: datetime = Field(..., description="Simulation timestamp")
    circuit_depth: int = Field(..., ge=0, description="Quantum circuit depth")
    num_qubits: int = Field(..., ge=1, description="Number of qubits")
    shots: int = Field(..., gt=0, description="Number of simulation shots")
    backend_name: str = Field(..., description="Simulation backend name")
    noise_model: Optional[Dict[str, Any]] = Field(None, description="Noise model parameters")
    execution_time: float = Field(..., ge=0.0, description="Execution time in seconds")
    results: Dict[str, Any] = Field(..., description="Simulation results")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        arbitrary_types_allowed = True


class AnalysisSchema(BaseModel):
    """Schema for analysis results."""
    
    analysis_id: str = Field(..., description="Unique analysis identifier")
    timestamp: datetime = Field(..., description="Analysis timestamp")
    analysis_type: str = Field(..., description="Type of analysis performed")
    input_data: List[str] = Field(..., description="Input data identifiers")
    parameters: Dict[str, Any] = Field(..., description="Analysis parameters")
    results: Dict[str, Any] = Field(..., description="Analysis results")
    confidence_intervals: Optional[Dict[str, List[float]]] = Field(None, description="Confidence intervals")
    statistical_tests: Optional[Dict[str, Any]] = Field(None, description="Statistical test results")
    visualizations: List[str] = Field(default_factory=list, description="Generated visualization paths")
    
    class Config:
        arbitrary_types_allowed = True


class ExperimentBatchSchema(BaseModel):
    """Schema for batch experiment results."""
    
    batch_id: str = Field(..., description="Unique batch identifier")
    timestamp: datetime = Field(..., description="Batch timestamp")
    experiment_count: int = Field(..., gt=0, description="Number of experiments in batch")
    experiments: List[TeleportationSchema] = Field(..., description="Individual experiments")
    batch_statistics: Dict[str, float] = Field(..., description="Batch-level statistics")
    batch_metadata: Dict[str, Any] = Field(default_factory=dict, description="Batch metadata")
    
    class Config:
        arbitrary_types_allowed = True


class ConfigurationSchema(BaseModel):
    """Schema for QuLab configuration."""
    
    config_id: str = Field(..., description="Configuration identifier")
    timestamp: datetime = Field(..., description="Configuration timestamp")
    version: str = Field(..., description="QuLab version")
    quantum_backend: str = Field(..., description="Default quantum backend")
    default_shots: int = Field(..., gt=0, description="Default number of shots")
    noise_model: Optional[Dict[str, Any]] = Field(None, description="Default noise model")
    governance_settings: Dict[str, Any] = Field(default_factory=dict, description="Governance settings")
    encoding_settings: Dict[str, Any] = Field(default_factory=dict, description="Encoding settings")
    output_settings: Dict[str, Any] = Field(default_factory=dict, description="Output settings")
    
    class Config:
        arbitrary_types_allowed = True


class ErrorSchema(BaseModel):
    """Schema for error reporting."""
    
    error_id: str = Field(..., description="Unique error identifier")
    timestamp: datetime = Field(..., description="Error timestamp")
    error_type: str = Field(..., description="Type of error")
    error_message: str = Field(..., description="Error message")
    stack_trace: Optional[str] = Field(None, description="Stack trace")
    context: Dict[str, Any] = Field(default_factory=dict, description="Error context")
    severity: str = Field(..., description="Error severity (low, medium, high, critical)")
    resolved: bool = Field(False, description="Whether error has been resolved")
    
    class Config:
        arbitrary_types_allowed = True


class ValidationResult(BaseModel):
    """Schema for validation results."""
    
    is_valid: bool = Field(..., description="Whether data is valid")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    schema_version: str = Field(..., description="Schema version used")
    validation_timestamp: datetime = Field(..., description="Validation timestamp")
    
    class Config:
        arbitrary_types_allowed = True
