"""
FastAPI endpoints for quantum teleportation.

Provides REST API endpoints for teleportation experiments with
comprehensive validation, error handling, and result serialization.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, validator
import uuid
import logging

from quantum_lab.protocols.teleportation import TeleportationProtocol, TeleportationResult
from ingest.schemas import TeleportationSchema
from ingest.results import ResultsManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/teleport", tags=["teleportation"])

# Global results manager
results_manager = ResultsManager()


class TeleportationRequest(BaseModel):
    """Request model for teleportation experiments."""
    
    alpha: float = Field(..., ge=0.0, le=1.0, description="Amplitude of |0⟩ state")
    beta: float = Field(..., ge=0.0, le=1.0, description="Amplitude of |1⟩ state")
    shots: int = Field(1024, ge=1, le=100000, description="Number of measurement shots")
    experiment_id: Optional[str] = Field(None, description="Custom experiment ID")
    save_results: bool = Field(True, description="Save results to storage")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('beta')
    def validate_normalization(cls, v, values):
        """Validate that quantum state is normalized."""
        if 'alpha' in values:
            alpha = values['alpha']
            import numpy as np
            if not np.isclose(abs(alpha)**2 + abs(v)**2, 1.0, atol=1e-10):
                raise ValueError("Quantum state must be normalized: |α|² + |β|² = 1")
        return v


class TeleportationResponse(BaseModel):
    """Response model for teleportation experiments."""
    
    experiment_id: str = Field(..., description="Experiment identifier")
    fidelity: float = Field(..., description="Teleportation fidelity")
    success_probability: float = Field(..., description="Success probability")
    execution_time: float = Field(..., description="Execution time in seconds")
    shots: int = Field(..., description="Number of shots")
    measurement_results: Dict[str, int] = Field(..., description="Bell measurement outcomes")
    classical_bits: List[int] = Field(..., description="Classical bits sent to Bob")
    status: str = Field("success", description="Experiment status")


class BatchTeleportationRequest(BaseModel):
    """Request model for batch teleportation experiments."""
    
    experiments: List[TeleportationRequest] = Field(..., description="List of experiments")
    batch_id: Optional[str] = Field(None, description="Custom batch ID")
    save_results: bool = Field(True, description="Save results to storage")


class BatchTeleportationResponse(BaseModel):
    """Response model for batch teleportation experiments."""
    
    batch_id: str = Field(..., description="Batch identifier")
    experiment_count: int = Field(..., description="Number of experiments")
    results: List[TeleportationResponse] = Field(..., description="Experiment results")
    batch_statistics: Dict[str, float] = Field(..., description="Batch-level statistics")
    total_execution_time: float = Field(..., description="Total execution time")


@router.post("/", response_model=TeleportationResponse)
async def teleport_quantum_state(request: TeleportationRequest):
    """
    Perform quantum teleportation experiment.
    
    Teleports an unknown quantum state |ψ⟩ = α|0⟩ + β|1⟩ from Alice to Bob
    using a shared Bell pair and classical communication.
    
    Args:
        request: Teleportation experiment parameters
        
    Returns:
        TeleportationResponse with experiment results
        
    Raises:
        HTTPException: If experiment fails or parameters are invalid
    """
    try:
        # Generate experiment ID if not provided
        experiment_id = request.experiment_id or str(uuid.uuid4())
        
        # Create teleportation protocol
        protocol = TeleportationProtocol()
        
        # Execute teleportation
        result = protocol.teleport(
            alpha=request.alpha,
            beta=request.beta,
            shots=request.shots
        )
        
        # Create response
        response = TeleportationResponse(
            experiment_id=experiment_id,
            fidelity=result.fidelity,
            success_probability=result.success_probability,
            execution_time=result.execution_time,
            shots=result.shots,
            measurement_results=result.measurement_results,
            classical_bits=list(result.classical_bits),
            status="success"
        )
        
        # Save results if requested
        if request.save_results:
            teleportation_schema = TeleportationSchema(
                experiment_id=experiment_id,
                timestamp=result.execution_time,  # Use execution time as timestamp
                alpha=request.alpha,
                beta=request.beta,
                fidelity=result.fidelity,
                success_probability=result.success_probability,
                shots=result.shots,
                execution_time=result.execution_time,
                measurement_results=result.measurement_results,
                classical_bits=list(result.classical_bits),
                metadata=request.metadata
            )
            
            results_manager.save_teleportation_result(teleportation_schema)
        
        logger.info(f"Teleportation experiment {experiment_id} completed successfully")
        return response
        
    except Exception as e:
        logger.error(f"Teleportation experiment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=BatchTeleportationResponse)
async def teleport_batch(request: BatchTeleportationRequest, background_tasks: BackgroundTasks):
    """
    Perform batch teleportation experiments.
    
    Executes multiple teleportation experiments in sequence with
    comprehensive batch-level statistics.
    
    Args:
        request: Batch teleportation parameters
        background_tasks: Background tasks for saving results
        
    Returns:
        BatchTeleportationResponse with batch results
    """
    try:
        batch_id = request.batch_id or str(uuid.uuid4())
        results = []
        total_execution_time = 0.0
        
        # Execute experiments
        for i, exp_request in enumerate(request.experiments):
            try:
                # Generate experiment ID
                experiment_id = exp_request.experiment_id or f"{batch_id}_{i}"
                
                # Create protocol
                protocol = TeleportationProtocol()
                
                # Execute teleportation
                result = protocol.teleport(
                    alpha=exp_request.alpha,
                    beta=exp_request.beta,
                    shots=exp_request.shots
                )
                
                # Create response
                response = TeleportationResponse(
                    experiment_id=experiment_id,
                    fidelity=result.fidelity,
                    success_probability=result.success_probability,
                    execution_time=result.execution_time,
                    shots=result.shots,
                    measurement_results=result.measurement_results,
                    classical_bits=list(result.classical_bits),
                    status="success"
                )
                
                results.append(response)
                total_execution_time += result.execution_time
                
                # Save individual result if requested
                if request.save_results:
                    teleportation_schema = TeleportationSchema(
                        experiment_id=experiment_id,
                        timestamp=result.execution_time,
                        alpha=exp_request.alpha,
                        beta=exp_request.beta,
                        fidelity=result.fidelity,
                        success_probability=result.success_probability,
                        shots=result.shots,
                        execution_time=result.execution_time,
                        measurement_results=result.measurement_results,
                        classical_bits=list(result.classical_bits),
                        metadata=exp_request.metadata
                    )
                    
                    background_tasks.add_task(
                        results_manager.save_teleportation_result, 
                        teleportation_schema
                    )
                
            except Exception as e:
                logger.error(f"Experiment {i} in batch {batch_id} failed: {e}")
                # Add failed experiment to results
                results.append(TeleportationResponse(
                    experiment_id=f"{batch_id}_{i}",
                    fidelity=0.0,
                    success_probability=0.0,
                    execution_time=0.0,
                    shots=exp_request.shots,
                    measurement_results={},
                    classical_bits=[],
                    status="failed"
                ))
        
        # Calculate batch statistics
        successful_results = [r for r in results if r.status == "success"]
        if successful_results:
            fidelities = [r.fidelity for r in successful_results]
            batch_statistics = {
                "mean_fidelity": sum(fidelities) / len(fidelities),
                "std_fidelity": (sum((f - sum(fidelities)/len(fidelities))**2 for f in fidelities) / len(fidelities))**0.5,
                "min_fidelity": min(fidelities),
                "max_fidelity": max(fidelities),
                "success_rate": len(successful_results) / len(results),
                "total_experiments": len(results),
                "successful_experiments": len(successful_results)
            }
        else:
            batch_statistics = {
                "mean_fidelity": 0.0,
                "std_fidelity": 0.0,
                "min_fidelity": 0.0,
                "max_fidelity": 0.0,
                "success_rate": 0.0,
                "total_experiments": len(results),
                "successful_experiments": 0
            }
        
        response = BatchTeleportationResponse(
            batch_id=batch_id,
            experiment_count=len(results),
            results=results,
            batch_statistics=batch_statistics,
            total_execution_time=total_execution_time
        )
        
        logger.info(f"Batch teleportation {batch_id} completed: {len(successful_results)}/{len(results)} successful")
        return response
        
    except Exception as e:
        logger.error(f"Batch teleportation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fidelity-bands")
async def analyze_fidelity_bands(alpha: float, beta: float, shots: int = 1024, 
                               num_trials: int = 100):
    """
    Analyze fidelity confidence bands using Monte Carlo sampling.
    
    Args:
        alpha: Amplitude of |0⟩ state
        beta: Amplitude of |1⟩ state
        shots: Number of shots per trial
        num_trials: Number of Monte Carlo trials
        
    Returns:
        Dictionary with fidelity statistics and confidence intervals
    """
    try:
        # Validate parameters
        if not 0 <= alpha <= 1 or not 0 <= beta <= 1:
            raise HTTPException(status_code=400, detail="Alpha and beta must be between 0 and 1")
        
        import numpy as np
        if not np.isclose(abs(alpha)**2 + abs(beta)**2, 1.0, atol=1e-10):
            raise HTTPException(status_code=400, detail="State must be normalized")
        
        if shots <= 0 or num_trials <= 0:
            raise HTTPException(status_code=400, detail="Shots and trials must be positive")
        
        # Create protocol and analyze
        protocol = TeleportationProtocol()
        bands = protocol.analyze_fidelity_bands(alpha, beta, shots, num_trials)
        
        return {
            "alpha": alpha,
            "beta": beta,
            "shots": shots,
            "num_trials": num_trials,
            "fidelity_bands": bands
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fidelity bands analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results/{experiment_id}")
async def get_teleportation_result(experiment_id: str):
    """
    Retrieve teleportation experiment result by ID.
    
    Args:
        experiment_id: Experiment identifier
        
    Returns:
        TeleportationSchema with experiment result
    """
    try:
        results = results_manager.load_teleportation_results(experiment_id=experiment_id)
        
        if not results:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        return results[0]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve experiment {experiment_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results")
async def list_teleportation_results(limit: int = 100, offset: int = 0):
    """
    List teleportation experiment results.
    
    Args:
        limit: Maximum number of results to return
        offset: Number of results to skip
        
    Returns:
        List of teleportation results
    """
    try:
        results = results_manager.load_teleportation_results()
        
        # Apply pagination
        start_idx = offset
        end_idx = offset + limit
        paginated_results = results[start_idx:end_idx]
        
        return {
            "results": paginated_results,
            "total_count": len(results),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Failed to list teleportation results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/results/{experiment_id}")
async def delete_teleportation_result(experiment_id: str):
    """
    Delete teleportation experiment result.
    
    Args:
        experiment_id: Experiment identifier
        
    Returns:
        Success message
    """
    try:
        # Note: This is a simplified implementation
        # In a real system, you'd implement proper deletion logic
        return {"message": f"Experiment {experiment_id} deletion requested"}
        
    except Exception as e:
        logger.error(f"Failed to delete experiment {experiment_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics")
async def get_teleportation_statistics():
    """
    Get teleportation experiment statistics.
    
    Returns:
        Dictionary with teleportation statistics
    """
    try:
        stats = results_manager.get_statistics("teleportation")
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get teleportation statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))
