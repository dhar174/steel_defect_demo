"""
Model serialization and persistence utilities for steel casting defect prediction models.
"""

import joblib
import pickle
import json
import yaml
import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from datetime import datetime
import logging
import hashlib


class ModelPersistence:
    """Handle model serialization, loading, and versioning"""
    
    def __init__(self, base_dir: str = "models/artifacts"):
        """
        Initialize model persistence manager
        
        Args:
            base_dir: Base directory for model storage
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def save_model(self,
                   model: Any,
                   model_name: str,
                   metadata: Optional[Dict[str, Any]] = None,
                   version: Optional[str] = None,
                   compress: bool = True,
                   format: str = 'joblib') -> str:
        """
        Save model with metadata and versioning
        
        Args:
            model: Model object to save
            model_name: Name identifier for the model
            metadata: Additional metadata to save with model
            version: Version string (auto-generated if None)
            compress: Whether to compress the saved model
            format: Serialization format ('joblib', 'pickle')
            
        Returns:
            Path to saved model
        """
        # Generate version if not provided
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create versioned directory
        model_dir = self.base_dir / model_name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare file paths
        if format == 'joblib':
            model_file = model_dir / "model.joblib"
        elif format == 'pickle':
            model_file = model_dir / "model.pkl"
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        metadata_file = model_dir / "metadata.json"
        
        try:
            # Save model
            if format == 'joblib':
                if compress:
                    joblib.dump(model, model_file, compress=3)
                else:
                    joblib.dump(model, model_file)
            elif format == 'pickle':
                with open(model_file, 'wb') as f:
                    pickle.dump(model, f)
            
            # Prepare metadata
            full_metadata = {
                'model_name': model_name,
                'version': version,
                'timestamp': datetime.now().isoformat(),
                'format': format,
                'compressed': compress,
                'file_size_mb': model_file.stat().st_size / (1024 * 1024),
                'model_hash': self._calculate_file_hash(model_file)
            }
            
            # Add user metadata
            if metadata:
                full_metadata.update(metadata)
            
            # Save metadata
            with open(metadata_file, 'w') as f:
                json.dump(full_metadata, f, indent=2, default=str)
            
            # Create symlink to latest version
            latest_link = self.base_dir / model_name / "latest"
            if latest_link.exists() or latest_link.is_symlink():
                latest_link.unlink()
            latest_link.symlink_to(version, target_is_directory=True)
            
            self.logger.info(f"Model saved: {model_file}")
            self.logger.info(f"Version: {version}")
            
            return str(model_file)
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            # Cleanup on failure
            if model_dir.exists():
                shutil.rmtree(model_dir)
            raise
    
    def load_model(self,
                   model_name: str,
                   version: str = "latest",
                   format: str = 'joblib') -> tuple:
        """
        Load model and metadata
        
        Args:
            model_name: Name identifier for the model
            version: Version to load ('latest' for most recent)
            format: Serialization format ('joblib', 'pickle')
            
        Returns:
            Tuple of (model, metadata)
        """
        # Resolve version
        if version == "latest":
            latest_link = self.base_dir / model_name / "latest"
            if not latest_link.exists():
                raise FileNotFoundError(f"No models found for {model_name}")
            model_dir = latest_link.resolve()
        else:
            model_dir = self.base_dir / model_name / version
            if not model_dir.exists():
                raise FileNotFoundError(f"Model version {version} not found for {model_name}")
        
        # Prepare file paths
        if format == 'joblib':
            model_file = model_dir / "model.joblib"
        elif format == 'pickle':
            model_file = model_dir / "model.pkl"
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        metadata_file = model_dir / "metadata.json"
        
        try:
            # Load model
            if format == 'joblib':
                model = joblib.load(model_file)
            elif format == 'pickle':
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
            
            # Load metadata
            metadata = {}
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            
            # Verify model integrity
            current_hash = self._calculate_file_hash(model_file)
            stored_hash = metadata.get('model_hash')
            if stored_hash and current_hash != stored_hash:
                self.logger.warning("Model file hash mismatch - file may be corrupted")
            
            self.logger.info(f"Model loaded: {model_file}")
            return model, metadata
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all available models
        
        Returns:
            List of model information dictionaries
        """
        models = []
        
        for model_path in self.base_dir.iterdir():
            if model_path.is_dir() and model_path.name != '.DS_Store':
                model_name = model_path.name
                
                # Get versions
                versions = []
                for version_path in model_path.iterdir():
                    if version_path.is_dir() and version_path.name != 'latest':
                        metadata_file = version_path / "metadata.json"
                        if metadata_file.exists():
                            try:
                                with open(metadata_file, 'r') as f:
                                    metadata = json.load(f)
                                versions.append(metadata)
                            except Exception:
                                pass
                
                if versions:
                    # Sort by timestamp
                    versions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
                    
                    models.append({
                        'model_name': model_name,
                        'versions': versions,
                        'latest_version': versions[0] if versions else None,
                        'total_versions': len(versions)
                    })
        
        return models
    
    def list_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """
        List all versions of a specific model
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of version metadata
        """
        model_path = self.base_dir / model_name
        if not model_path.exists():
            return []
        
        versions = []
        for version_path in model_path.iterdir():
            if version_path.is_dir() and version_path.name != 'latest':
                metadata_file = version_path / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        versions.append(metadata)
                    except Exception:
                        pass
        
        # Sort by timestamp
        versions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return versions
    
    def delete_model(self, model_name: str, version: Optional[str] = None) -> None:
        """
        Delete model or specific version
        
        Args:
            model_name: Name of the model
            version: Specific version to delete (None deletes all versions)
        """
        if version:
            # Delete specific version
            version_path = self.base_dir / model_name / version
            if version_path.exists():
                shutil.rmtree(version_path)
                self.logger.info(f"Deleted model version: {model_name}/{version}")
                
                # Update latest symlink if necessary
                latest_link = self.base_dir / model_name / "latest"
                if latest_link.exists() and latest_link.resolve().name == version:
                    latest_link.unlink()
                    # Find next latest version
                    remaining_versions = self.list_versions(model_name)
                    if remaining_versions:
                        next_latest = remaining_versions[0]['version']
                        latest_link.symlink_to(next_latest, target_is_directory=True)
        else:
            # Delete all versions
            model_path = self.base_dir / model_name
            if model_path.exists():
                shutil.rmtree(model_path)
                self.logger.info(f"Deleted all versions of model: {model_name}")
    
    def export_model(self,
                     model_name: str,
                     export_path: str,
                     version: str = "latest",
                     include_metadata: bool = True) -> None:
        """
        Export model to external location
        
        Args:
            model_name: Name of the model
            export_path: Path to export to
            version: Version to export
            include_metadata: Whether to include metadata
        """
        # Load model and metadata
        model, metadata = self.load_model(model_name, version)
        
        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Export model
        model_format = metadata.get('format', 'joblib')
        if model_format == 'joblib':
            model_file = export_dir / f"{model_name}.joblib"
            joblib.dump(model, model_file)
        elif model_format == 'pickle':
            model_file = export_dir / f"{model_name}.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
        
        # Export metadata if requested
        if include_metadata:
            metadata_file = export_dir / f"{model_name}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        
        self.logger.info(f"Model exported to: {export_path}")
    
    def import_model(self,
                     import_path: str,
                     model_name: str,
                     version: Optional[str] = None) -> None:
        """
        Import model from external location
        
        Args:
            import_path: Path to import from
            model_name: Name to assign to imported model
            version: Version to assign (auto-generated if None)
        """
        import_dir = Path(import_path)
        if not import_dir.exists():
            raise FileNotFoundError(f"Import path does not exist: {import_path}")
        
        # Find model file
        model_file = None
        for ext in ['.joblib', '.pkl']:
            potential_file = import_dir / f"{model_name}{ext}"
            if potential_file.exists():
                model_file = potential_file
                break
        
        if model_file is None:
            raise FileNotFoundError("No model file found in import directory")
        
        # Load model
        if model_file.suffix == '.joblib':
            model = joblib.load(model_file)
            format = 'joblib'
        elif model_file.suffix == '.pkl':
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            format = 'pickle'
        
        # Load metadata if available
        metadata_file = import_dir / f"{model_name}_metadata.json"
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        
        # Add import information to metadata
        metadata['imported_from'] = str(import_path)
        metadata['import_timestamp'] = datetime.now().isoformat()
        
        # Save imported model
        self.save_model(model, model_name, metadata, version, format=format)
        self.logger.info(f"Model imported: {model_name}")
    
    def cleanup_old_versions(self,
                           model_name: str,
                           keep_versions: int = 5) -> None:
        """
        Clean up old model versions, keeping only the most recent ones
        
        Args:
            model_name: Name of the model
            keep_versions: Number of versions to keep
        """
        versions = self.list_versions(model_name)
        
        if len(versions) <= keep_versions:
            return
        
        # Delete old versions
        versions_to_delete = versions[keep_versions:]
        for version_info in versions_to_delete:
            version = version_info['version']
            self.delete_model(model_name, version)
            
        self.logger.info(f"Cleaned up {len(versions_to_delete)} old versions of {model_name}")
    
    def get_model_size(self, model_name: str, version: str = "latest") -> float:
        """
        Get model file size in MB
        
        Args:
            model_name: Name of the model
            version: Version to check
            
        Returns:
            Model size in MB
        """
        _, metadata = self.load_model(model_name, version)
        return metadata.get('file_size_mb', 0.0)
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def get_storage_summary(self) -> Dict[str, Any]:
        """
        Get summary of storage usage
        
        Returns:
            Dictionary with storage information
        """
        models = self.list_models()
        total_size = 0
        total_models = len(models)
        total_versions = 0
        
        for model_info in models:
            total_versions += model_info['total_versions']
            for version in model_info['versions']:
                total_size += version.get('file_size_mb', 0)
        
        return {
            'total_models': total_models,
            'total_versions': total_versions,
            'total_size_mb': total_size,
            'storage_path': str(self.base_dir),
            'models': models
        }