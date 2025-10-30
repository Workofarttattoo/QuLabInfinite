"""
Results management for QuLab experiments.

Handles saving, loading, and managing experimental results with support
for multiple formats (JSON, Parquet, CSV) and efficient data storage.
"""

from typing import List, Dict, Optional, Union, Any
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

from .schemas import (
    TeleportationSchema, GovernanceSchema, EncodingSchema, 
    SimulationSchema, AnalysisSchema, ExperimentBatchSchema
)

logger = logging.getLogger(__name__)


class ExperimentResult:
    """Container for individual experiment results."""
    
    def __init__(self, data: Dict[str, Any], schema_type: str):
        """
        Initialize experiment result.
        
        Args:
            data: Experiment data
            schema_type: Type of schema ('teleportation', 'governance', 'encoding', etc.)
        """
        self.data = data
        self.schema_type = schema_type
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'schema_type': self.schema_type,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        # Flatten nested data for DataFrame
        flattened = self._flatten_dict(self.data)
        return pd.DataFrame([flattened])
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Convert lists to strings for DataFrame compatibility
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)


class ResultsManager:
    """
    Manager for experimental results storage and retrieval.
    
    Supports multiple storage formats and provides efficient querying
    and analysis capabilities.
    """
    
    def __init__(self, base_path: Union[str, Path] = "results"):
        """
        Initialize results manager.
        
        Args:
            base_path: Base directory for storing results
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different data types
        self.teleportation_path = self.base_path / "teleportation"
        self.governance_path = self.base_path / "governance"
        self.encoding_path = self.base_path / "encoding"
        self.simulation_path = self.base_path / "simulation"
        self.analysis_path = self.base_path / "analysis"
        
        for path in [self.teleportation_path, self.governance_path, 
                    self.encoding_path, self.simulation_path, self.analysis_path]:
            path.mkdir(exist_ok=True)
    
    def save_teleportation_result(self, result: TeleportationSchema, 
                                format: str = 'json') -> str:
        """
        Save teleportation experiment result.
        
        Args:
            result: Teleportation result to save
            format: Storage format ('json', 'parquet', 'csv')
            
        Returns:
            Path to saved file
        """
        return self._save_result(result, self.teleportation_path, format)
    
    def save_governance_result(self, result: GovernanceSchema, 
                             format: str = 'json') -> str:
        """
        Save governance/evidence result.
        
        Args:
            result: Governance result to save
            format: Storage format ('json', 'parquet', 'csv')
            
        Returns:
            Path to saved file
        """
        return self._save_result(result, self.governance_path, format)
    
    def save_encoding_result(self, result: EncodingSchema, 
                           format: str = 'json') -> str:
        """
        Save encoding operation result.
        
        Args:
            result: Encoding result to save
            format: Storage format ('json', 'parquet', 'csv')
            
        Returns:
            Path to saved file
        """
        return self._save_result(result, self.encoding_path, format)
    
    def save_simulation_result(self, result: SimulationSchema, 
                             format: str = 'json') -> str:
        """
        Save simulation result.
        
        Args:
            result: Simulation result to save
            format: Storage format ('json', 'parquet', 'csv')
            
        Returns:
            Path to saved file
        """
        return self._save_result(result, self.simulation_path, format)
    
    def save_analysis_result(self, result: AnalysisSchema, 
                           format: str = 'json') -> str:
        """
        Save analysis result.
        
        Args:
            result: Analysis result to save
            format: Storage format ('json', 'parquet', 'csv')
            
        Returns:
            Path to saved file
        """
        return self._save_result(result, self.analysis_path, format)
    
    def _save_result(self, result: Any, directory: Path, format: str) -> str:
        """Save result in specified format."""
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if hasattr(result, 'experiment_id'):
            filename = f"{result.experiment_id}_{timestamp}"
        elif hasattr(result, 'ledger_id'):
            filename = f"{result.ledger_id}_{timestamp}"
        elif hasattr(result, 'operation_id'):
            filename = f"{result.operation_id}_{timestamp}"
        elif hasattr(result, 'simulation_id'):
            filename = f"{result.simulation_id}_{timestamp}"
        elif hasattr(result, 'analysis_id'):
            filename = f"{result.analysis_id}_{timestamp}"
        else:
            filename = f"result_{timestamp}"
        
        if format == 'json':
            filepath = directory / f"{filename}.json"
            with open(filepath, 'w') as f:
                json.dump(result.dict(), f, indent=2, default=str)
        elif format == 'parquet':
            filepath = directory / f"{filename}.parquet"
            df = pd.DataFrame([result.dict()])
            df.to_parquet(filepath, index=False)
        elif format == 'csv':
            filepath = directory / f"{filename}.csv"
            df = pd.DataFrame([result.dict()])
            df.to_csv(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved result to {filepath}")
        return str(filepath)
    
    def load_teleportation_results(self, experiment_id: Optional[str] = None,
                                 start_date: Optional[datetime] = None,
                                 end_date: Optional[datetime] = None) -> List[TeleportationSchema]:
        """
        Load teleportation results with optional filtering.
        
        Args:
            experiment_id: Filter by experiment ID
            start_date: Filter by start date
            end_date: Filter by end date
            
        Returns:
            List of teleportation results
        """
        return self._load_results(self.teleportation_path, TeleportationSchema, 
                                experiment_id, start_date, end_date)
    
    def load_governance_results(self, ledger_id: Optional[str] = None,
                              start_date: Optional[datetime] = None,
                              end_date: Optional[datetime] = None) -> List[GovernanceSchema]:
        """
        Load governance results with optional filtering.
        
        Args:
            ledger_id: Filter by ledger ID
            start_date: Filter by start date
            end_date: Filter by end date
            
        Returns:
            List of governance results
        """
        return self._load_results(self.governance_path, GovernanceSchema,
                                ledger_id, start_date, end_date)
    
    def load_encoding_results(self, operation_id: Optional[str] = None,
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None) -> List[EncodingSchema]:
        """
        Load encoding results with optional filtering.
        
        Args:
            operation_id: Filter by operation ID
            start_date: Filter by start date
            end_date: Filter by end date
            
        Returns:
            List of encoding results
        """
        return self._load_results(self.encoding_path, EncodingSchema,
                                operation_id, start_date, end_date)
    
    def _load_results(self, directory: Path, schema_class: Any, 
                     id_filter: Optional[str] = None,
                     start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None) -> List[Any]:
        """Load results from directory with filtering."""
        results = []
        
        for filepath in directory.glob("*.json"):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # Apply filters
                if id_filter and id_filter not in str(data):
                    continue
                
                if start_date or end_date:
                    timestamp_str = data.get('timestamp', '')
                    if timestamp_str:
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        if start_date and timestamp < start_date:
                            continue
                        if end_date and timestamp > end_date:
                            continue
                
                result = schema_class(**data)
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Failed to load {filepath}: {e}")
        
        return results
    
    def get_results_dataframe(self, result_type: str, 
                            filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Get results as pandas DataFrame.
        
        Args:
            result_type: Type of results ('teleportation', 'governance', 'encoding', etc.)
            filters: Optional filters to apply
            
        Returns:
            DataFrame with results
        """
        if result_type == 'teleportation':
            results = self.load_teleportation_results()
        elif result_type == 'governance':
            results = self.load_governance_results()
        elif result_type == 'encoding':
            results = self.load_encoding_results()
        else:
            raise ValueError(f"Unknown result type: {result_type}")
        
        if not results:
            return pd.DataFrame()
        
        # Convert to DataFrame
        data = [result.dict() for result in results]
        df = pd.DataFrame(data)
        
        # Apply filters
        if filters:
            for key, value in filters.items():
                if key in df.columns:
                    df = df[df[key] == value]
        
        return df
    
    def export_results(self, result_type: str, output_path: str, 
                      format: str = 'csv', filters: Optional[Dict[str, Any]] = None):
        """
        Export results to file.
        
        Args:
            result_type: Type of results to export
            output_path: Output file path
            format: Export format ('csv', 'parquet', 'json')
            filters: Optional filters to apply
        """
        df = self.get_results_dataframe(result_type, filters)
        
        if format == 'csv':
            df.to_csv(output_path, index=False)
        elif format == 'parquet':
            df.to_parquet(output_path, index=False)
        elif format == 'json':
            df.to_json(output_path, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Exported {len(df)} results to {output_path}")
    
    def get_statistics(self, result_type: str) -> Dict[str, Any]:
        """
        Get statistics for results.
        
        Args:
            result_type: Type of results to analyze
            
        Returns:
            Dictionary with statistics
        """
        df = self.get_results_dataframe(result_type)
        
        if df.empty:
            return {}
        
        stats = {
            'total_count': len(df),
            'date_range': {
                'start': df['timestamp'].min() if 'timestamp' in df.columns else None,
                'end': df['timestamp'].max() if 'timestamp' in df.columns else None
            }
        }
        
        # Add type-specific statistics
        if result_type == 'teleportation' and 'fidelity' in df.columns:
            stats['fidelity'] = {
                'mean': df['fidelity'].mean(),
                'std': df['fidelity'].std(),
                'min': df['fidelity'].min(),
                'max': df['fidelity'].max(),
                'median': df['fidelity'].median()
            }
        
        if result_type == 'encoding' and 'compression_ratio' in df.columns:
            stats['compression'] = {
                'mean_ratio': df['compression_ratio'].mean(),
                'best_ratio': df['compression_ratio'].max(),
                'worst_ratio': df['compression_ratio'].min()
            }
        
        return stats
    
    def cleanup_old_results(self, days: int = 30):
        """
        Clean up old result files.
        
        Args:
            days: Number of days to keep results
        """
        cutoff_date = datetime.now() - pd.Timedelta(days=days)
        deleted_count = 0
        
        for directory in [self.teleportation_path, self.governance_path, 
                         self.encoding_path, self.simulation_path, self.analysis_path]:
            for filepath in directory.glob("*"):
                if filepath.is_file():
                    file_time = datetime.fromtimestamp(filepath.stat().st_mtime)
                    if file_time < cutoff_date:
                        filepath.unlink()
                        deleted_count += 1
        
        logger.info(f"Cleaned up {deleted_count} old result files")
    
    def backup_results(self, backup_path: Union[str, Path]):
        """
        Create backup of all results.
        
        Args:
            backup_path: Path for backup
        """
        import shutil
        
        backup_path = Path(backup_path)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = backup_path / f"qulab_backup_{timestamp}"
        
        shutil.copytree(self.base_path, backup_dir)
        logger.info(f"Created backup at {backup_dir}")
