"""Model artifact management for training pipeline"""

import json
import pickle
import joblib
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List, Union, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import yaml
import hashlib
import zipfile


@dataclass
class ArtifactMetadata:
    """Metadata for model artifacts"""
    artifact_type: str
    created_at: str
    model_name: str
    experiment_name: str
    version: str
    file_path: str
    file_size: int
    checksum: str
    metrics: Dict[str, Any] = None
    config: Dict[str, Any] = None
    dependencies: List[str] = None
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class ArtifactManager:
    """Manage model artifacts and results"""
    
    def __init__(self, 
                 output_dir: str,
                 experiment_name: Optional[str] = None,
                 versioning: bool = True,
                 compression: bool = True):
        """
        Initialize artifact manager
        
        Args:
            output_dir: Base output directory for artifacts
            experiment_name: Optional experiment name for organization
            versioning: Enable automatic versioning
            compression: Enable compression for large artifacts
        """
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name or "default"
        self.versioning = versioning
        self.compression = compression
        
        # Create directory structure
        self._setup_directories()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Track artifacts
        self.artifacts: Dict[str, ArtifactMetadata] = {}
        self.artifact_index_file = self.output_dir / "artifact_index.json"
        
        # Load existing artifact index
        self._load_artifact_index()
        
        self.logger.info(f"Artifact manager initialized: {self.output_dir}")
    
    def _setup_directories(self) -> None:
        """Setup directory structure"""
        dirs = [
            self.output_dir,
            self.output_dir / "models",
            self.output_dir / "feature_engineers", 
            self.output_dir / "results",
            self.output_dir / "configs",
            self.output_dir / "plots",
            self.output_dir / "logs",
            self.output_dir / "checkpoints",
            self.output_dir / "metadata"
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _load_artifact_index(self) -> None:
        """Load existing artifact index"""
        if self.artifact_index_file.exists():
            try:
                with open(self.artifact_index_file, 'r') as f:
                    index_data = json.load(f)
                    
                for artifact_id, metadata_dict in index_data.items():
                    self.artifacts[artifact_id] = ArtifactMetadata(**metadata_dict)
                    
                self.logger.info(f"Loaded {len(self.artifacts)} existing artifacts")
            except Exception as e:
                self.logger.warning(f"Could not load artifact index: {e}")
    
    def _save_artifact_index(self) -> None:
        """Save artifact index to file"""
        try:
            index_data = {
                artifact_id: metadata.to_dict() 
                for artifact_id, metadata in self.artifacts.items()
            }
            
            with open(self.artifact_index_file, 'w') as f:
                json.dump(index_data, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Could not save artifact index: {e}")
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum of file"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(HASH_CHUNK_SIZE), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            self.logger.warning(f"Could not calculate checksum for {file_path}: {e}")
            return ""
    
    def _create_version(self, base_name: str, metrics: Optional[Dict[str, float]] = None) -> str:
        """Create version string for artifact"""
        if not self.versioning:
            return "latest"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if metrics and 'roc_auc' in metrics:
            auc_score = metrics['roc_auc']
            performance_str = f"auc{auc_score:.3f}".replace('.', '')
            return f"{base_name}_{timestamp}_{performance_str}"
        
        return f"{base_name}_{timestamp}"
    
    def save_model(self, 
                   model: Any, 
                   filename: str,
                   metadata: Optional[Dict[str, Any]] = None,
                   metrics: Optional[Dict[str, float]] = None) -> Path:
        """
        Save trained model
        
        Args:
            model: Trained model object
            filename: Base filename (without extension)
            metadata: Optional metadata dictionary
            metrics: Optional performance metrics
            
        Returns:
            Path to saved model file
        """
        # Determine file extension and save method
        if hasattr(model, 'save_model') and callable(model.save_model):
            # Use model's own save method
            file_path = self.output_dir / "models" / f"{filename}.pkl"
            model.save_model(str(file_path), include_metadata=True, compress=self.compression)
        else:
            # Use joblib for general objects
            file_path = self.output_dir / "models" / f"{filename}.joblib"
            if self.compression:
                joblib.dump(model, file_path, compress=3)
            else:
                joblib.dump(model, file_path)
        
        # Create artifact metadata
        version = self._create_version(filename, metrics)
        checksum = self._calculate_checksum(file_path)
        
        artifact_metadata = ArtifactMetadata(
            artifact_type="model",
            created_at=datetime.now().isoformat(),
            model_name=filename,
            experiment_name=self.experiment_name,
            version=version,
            file_path=str(file_path),
            file_size=file_path.stat().st_size,
            checksum=checksum,
            metrics=metrics,
            config=metadata,
            description=f"Trained model: {filename}"
        )
        
        # Save metadata
        self.artifacts[f"model_{filename}"] = artifact_metadata
        self._save_artifact_index()
        
        # Save separate metadata file
        metadata_path = self.output_dir / "metadata" / f"{filename}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(artifact_metadata.to_dict(), f, indent=2, default=str)
        
        self.logger.info(f"Model saved: {file_path} (size: {file_path.stat().st_size} bytes)")
        return file_path
    
    def save_feature_engineer(self, 
                              feature_engineer: Any, 
                              filename: str,
                              metadata: Optional[Dict[str, Any]] = None) -> Path:
        """
        Save feature engineering pipeline
        
        Args:
            feature_engineer: Feature engineering object
            filename: Base filename
            metadata: Optional metadata
            
        Returns:
            Path to saved feature engineer file
        """
        file_path = self.output_dir / "feature_engineers" / f"{filename}.joblib"
        
        if self.compression:
            joblib.dump(feature_engineer, file_path, compress=3)
        else:
            joblib.dump(feature_engineer, file_path)
        
        # Create artifact metadata
        version = self._create_version(filename)
        checksum = self._calculate_checksum(file_path)
        
        artifact_metadata = ArtifactMetadata(
            artifact_type="feature_engineer",
            created_at=datetime.now().isoformat(),
            model_name=filename,
            experiment_name=self.experiment_name,
            version=version,
            file_path=str(file_path),
            file_size=file_path.stat().st_size,
            checksum=checksum,
            config=metadata,
            description=f"Feature engineering pipeline: {filename}"
        )
        
        self.artifacts[f"feature_engineer_{filename}"] = artifact_metadata
        self._save_artifact_index()
        
        self.logger.info(f"Feature engineer saved: {file_path}")
        return file_path
    
    def save_results(self, 
                     results: Dict[str, Any], 
                     filename: str,
                     format: str = 'json') -> Path:
        """
        Save training results
        
        Args:
            results: Results dictionary
            filename: Base filename
            format: Output format ('json', 'yaml', 'pickle')
            
        Returns:
            Path to saved results file
        """
        if format == 'json':
            file_path = self.output_dir / "results" / f"{filename}.json"
            with open(file_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        elif format == 'yaml':
            file_path = self.output_dir / "results" / f"{filename}.yaml"
            with open(file_path, 'w') as f:
                yaml.dump(results, f, default_flow_style=False, indent=2)
        elif format == 'pickle':
            file_path = self.output_dir / "results" / f"{filename}.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(results, f)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Create artifact metadata
        version = self._create_version(filename)
        checksum = self._calculate_checksum(file_path)
        
        artifact_metadata = ArtifactMetadata(
            artifact_type="results",
            created_at=datetime.now().isoformat(),
            model_name=filename,
            experiment_name=self.experiment_name,
            version=version,
            file_path=str(file_path),
            file_size=file_path.stat().st_size,
            checksum=checksum,
            description=f"Training results: {filename}"
        )
        
        self.artifacts[f"results_{filename}"] = artifact_metadata
        self._save_artifact_index()
        
        self.logger.info(f"Results saved: {file_path}")
        return file_path
    
    def save_config(self, 
                    config: Dict[str, Any], 
                    filename: str,
                    format: str = 'yaml') -> Path:
        """
        Save training configuration
        
        Args:
            config: Configuration dictionary
            filename: Base filename
            format: Output format ('yaml', 'json')
            
        Returns:
            Path to saved config file
        """
        if format == 'yaml':
            file_path = self.output_dir / "configs" / f"{filename}.yaml"
            with open(file_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
        elif format == 'json':
            file_path = self.output_dir / "configs" / f"{filename}.json"
            with open(file_path, 'w') as f:
                json.dump(config, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Create artifact metadata
        version = self._create_version(filename)
        checksum = self._calculate_checksum(file_path)
        
        artifact_metadata = ArtifactMetadata(
            artifact_type="config",
            created_at=datetime.now().isoformat(),
            model_name=filename,
            experiment_name=self.experiment_name,
            version=version,
            file_path=str(file_path),
            file_size=file_path.stat().st_size,
            checksum=checksum,
            description=f"Training configuration: {filename}"
        )
        
        self.artifacts[f"config_{filename}"] = artifact_metadata
        self._save_artifact_index()
        
        self.logger.info(f"Configuration saved: {file_path}")
        return file_path
    
    def save_plot(self, 
                  figure: Any, 
                  filename: str,
                  format: str = 'png',
                  dpi: int = 300) -> Path:
        """
        Save matplotlib figure
        
        Args:
            figure: Matplotlib figure
            filename: Base filename
            format: Image format
            dpi: Image resolution
            
        Returns:
            Path to saved plot file
        """
        file_path = self.output_dir / "plots" / f"{filename}.{format}"
        
        try:
            figure.savefig(file_path, dpi=dpi, bbox_inches='tight', format=format)
            
            # Create artifact metadata
            version = self._create_version(filename)
            checksum = self._calculate_checksum(file_path)
            
            artifact_metadata = ArtifactMetadata(
                artifact_type="plot",
                created_at=datetime.now().isoformat(),
                model_name=filename,
                experiment_name=self.experiment_name,
                version=version,
                file_path=str(file_path),
                file_size=file_path.stat().st_size,
                checksum=checksum,
                description=f"Plot: {filename}"
            )
            
            self.artifacts[f"plot_{filename}"] = artifact_metadata
            self._save_artifact_index()
            
            self.logger.info(f"Plot saved: {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving plot {filename}: {e}")
            raise
        
        return file_path
    
    def save_checkpoint(self, 
                        model: Any,
                        optimizer_state: Optional[Dict] = None,
                        epoch: int = 0,
                        metrics: Optional[Dict[str, float]] = None) -> Path:
        """
        Save training checkpoint
        
        Args:
            model: Model object
            optimizer_state: Optimizer state dictionary
            epoch: Current epoch
            metrics: Current metrics
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_data = {
            'model': model,
            'optimizer_state': optimizer_state,
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        filename = f"checkpoint_epoch_{epoch:04d}"
        file_path = self.output_dir / "checkpoints" / f"{filename}.pkl"
        
        with open(file_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        self.logger.info(f"Checkpoint saved: {file_path}")
        return file_path
    
    def load_model(self, model_path: str) -> Any:
        """
        Load saved model
        
        Args:
            model_path: Path to saved model
            
        Returns:
            Loaded model object
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            if model_path.suffix == '.pkl':
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            elif model_path.suffix == '.joblib':
                model = joblib.load(model_path)
            else:
                raise ValueError(f"Unsupported model file format: {model_path.suffix}")
                
            self.logger.info(f"Model loaded from: {model_path}")
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model from {model_path}: {e}")
            raise
    
    def load_feature_engineer(self, fe_path: str) -> Any:
        """
        Load saved feature engineer
        
        Args:
            fe_path: Path to saved feature engineer
            
        Returns:
            Loaded feature engineer object
        """
        fe_path = Path(fe_path)
        
        if not fe_path.exists():
            raise FileNotFoundError(f"Feature engineer file not found: {fe_path}")
        
        try:
            feature_engineer = joblib.load(fe_path)
            self.logger.info(f"Feature engineer loaded from: {fe_path}")
            return feature_engineer
            
        except Exception as e:
            self.logger.error(f"Error loading feature engineer from {fe_path}: {e}")
            raise
    
    def load_results(self, results_path: str) -> Dict[str, Any]:
        """
        Load saved results
        
        Args:
            results_path: Path to saved results
            
        Returns:
            Loaded results dictionary
        """
        results_path = Path(results_path)
        
        if not results_path.exists():
            raise FileNotFoundError(f"Results file not found: {results_path}")
        
        try:
            if results_path.suffix == '.json':
                with open(results_path, 'r') as f:
                    results = json.load(f)
            elif results_path.suffix in ['.yaml', '.yml']:
                with open(results_path, 'r') as f:
                    results = yaml.safe_load(f)
            elif results_path.suffix == '.pkl':
                with open(results_path, 'rb') as f:
                    results = pickle.load(f)
            else:
                raise ValueError(f"Unsupported results file format: {results_path.suffix}")
                
            self.logger.info(f"Results loaded from: {results_path}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error loading results from {results_path}: {e}")
            raise
    
    def create_model_version(self, metrics: Dict[str, float]) -> str:
        """
        Create versioned model identifier
        
        Args:
            metrics: Model performance metrics
            
        Returns:
            Versioned model identifier
        """
        return self._create_version("model", metrics)
    
    def list_artifacts(self, artifact_type: Optional[str] = None) -> List[ArtifactMetadata]:
        """
        List all artifacts of specified type
        
        Args:
            artifact_type: Optional filter by artifact type
            
        Returns:
            List of artifact metadata
        """
        artifacts = list(self.artifacts.values())
        
        if artifact_type:
            artifacts = [a for a in artifacts if a.artifact_type == artifact_type]
        
        # Sort by creation time (newest first)
        artifacts.sort(key=lambda x: x.created_at, reverse=True)
        return artifacts
    
    def get_artifact(self, artifact_id: str) -> Optional[ArtifactMetadata]:
        """
        Get artifact metadata by ID
        
        Args:
            artifact_id: Artifact identifier
            
        Returns:
            Artifact metadata or None if not found
        """
        return self.artifacts.get(artifact_id)
    
    def delete_artifact(self, artifact_id: str, delete_file: bool = True) -> bool:
        """
        Delete artifact
        
        Args:
            artifact_id: Artifact identifier
            delete_file: Whether to delete the physical file
            
        Returns:
            True if successful
        """
        if artifact_id not in self.artifacts:
            self.logger.warning(f"Artifact not found: {artifact_id}")
            return False
        
        artifact = self.artifacts[artifact_id]
        
        # Delete physical file if requested
        if delete_file:
            file_path = Path(artifact.file_path)
            if file_path.exists():
                try:
                    file_path.unlink()
                    self.logger.info(f"Deleted file: {file_path}")
                except Exception as e:
                    self.logger.error(f"Error deleting file {file_path}: {e}")
                    return False
        
        # Remove from index
        del self.artifacts[artifact_id]
        self._save_artifact_index()
        
        self.logger.info(f"Artifact deleted: {artifact_id}")
        return True
    
    def export_artifacts(self, 
                        output_path: str,
                        artifact_types: Optional[List[str]] = None,
                        include_metadata: bool = True) -> Path:
        """
        Export artifacts to zip file
        
        Args:
            output_path: Path for output zip file
            artifact_types: Optional filter by artifact types
            include_metadata: Include metadata files
            
        Returns:
            Path to created zip file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        artifacts_to_export = self.list_artifacts()
        if artifact_types:
            artifacts_to_export = [a for a in artifacts_to_export if a.artifact_type in artifact_types]
        
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for artifact in artifacts_to_export:
                file_path = Path(artifact.file_path)
                if file_path.exists():
                    # Add file to zip with relative path
                    arcname = f"{artifact.artifact_type}/{file_path.name}"
                    zipf.write(file_path, arcname)
                    
                    # Add metadata if requested
                    if include_metadata:
                        metadata_content = json.dumps(artifact.to_dict(), indent=2, default=str)
                        metadata_arcname = f"metadata/{file_path.stem}_metadata.json"
                        zipf.writestr(metadata_arcname, metadata_content)
            
            # Add artifact index
            if include_metadata:
                index_content = json.dumps(
                    {aid: metadata.to_dict() for aid, metadata in self.artifacts.items()},
                    indent=2, default=str
                )
                zipf.writestr("artifact_index.json", index_content)
        
        self.logger.info(f"Artifacts exported to: {output_path}")
        return output_path
    
    def cleanup_old_artifacts(self, 
                             keep_count: int = 5,
                             artifact_type: Optional[str] = None) -> int:
        """
        Clean up old artifacts, keeping only the most recent ones
        
        Args:
            keep_count: Number of artifacts to keep
            artifact_type: Optional filter by artifact type
            
        Returns:
            Number of artifacts deleted
        """
        artifacts = self.list_artifacts(artifact_type)
        
        if len(artifacts) <= keep_count:
            return 0
        
        artifacts_to_delete = artifacts[keep_count:]
        deleted_count = 0
        
        for artifact in artifacts_to_delete:
            artifact_id = None
            for aid, ameta in self.artifacts.items():
                if ameta.file_path == artifact.file_path:
                    artifact_id = aid
                    break
            
            if artifact_id and self.delete_artifact(artifact_id, delete_file=True):
                deleted_count += 1
        
        self.logger.info(f"Cleaned up {deleted_count} old artifacts")
        return deleted_count
    
    def get_storage_summary(self) -> Dict[str, Any]:
        """
        Get storage usage summary
        
        Returns:
            Storage summary dictionary
        """
        artifacts = list(self.artifacts.values())
        
        total_size = sum(a.file_size for a in artifacts)
        type_counts = {}
        type_sizes = {}
        
        for artifact in artifacts:
            artifact_type = artifact.artifact_type
            type_counts[artifact_type] = type_counts.get(artifact_type, 0) + 1
            type_sizes[artifact_type] = type_sizes.get(artifact_type, 0) + artifact.file_size
        
        return {
            "total_artifacts": len(artifacts),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "by_type": {
                artifact_type: {
                    "count": type_counts[artifact_type],
                    "size_bytes": type_sizes[artifact_type],
                    "size_mb": type_sizes[artifact_type] / (1024 * 1024)
                }
                for artifact_type in type_counts.keys()
            }
        }