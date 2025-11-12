"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

__init__ - Part of Oncology Lab
"""

"""
QuLabInfinite Oncology Laboratory package.

Exports convenience imports for the tumour, drug, and intervention simulation
components that underpin the interactive lab. The toolkit is intended for
educational and prototyping scenarios and does not constitute medical advice.
"""

from .oncology_lab import (
    OncologyLaboratory,
    OncologyLabConfig,
    TumorType,
    CancerStage
)

from .tumor_simulator import (
    TumorSimulator,
    TumorGrowthModel,
    CancerCell,
    CellCyclePhase,
    TumorMicroenvironment
)

from .drug_response import (
    DrugSimulator,
    Drug,
    DrugClass,
    PharmacokineticModel
)

from .ten_field_controller import (
    TenFieldController,
    FieldInterventionProtocol,
    InterventionResponse
)

__all__ = [
    'OncologyLaboratory',
    'OncologyLabConfig',
    'TumorType',
    'CancerStage',
    'TumorSimulator',
    'TumorGrowthModel',
    'CancerCell',
    'CellCyclePhase',
    'TumorMicroenvironment',
    'DrugSimulator',
    'Drug',
    'DrugClass',
    'PharmacokineticModel',
    'TenFieldController',
    'FieldInterventionProtocol',
    'InterventionResponse',
]

__version__ = '1.0.0'
