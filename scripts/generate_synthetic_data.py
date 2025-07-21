#!/usr/bin/env python3
"""
Comprehensive synthetic steel casting data generation script.

This script provides a complete command-line interface for generating synthetic steel casting
time-series data with realistic sensor patterns, controlled defect scenarios, and comprehensive
data quality validation.

Features:
- Configurable data generation parameters
- Real-time progress tracking with ETA
- Comprehensive logging and performance monitoring
- Data quality validation
- Organized output file structure
- Error handling and recovery
"""

import sys
import argparse
import time
from pathlib import Path
from typing import Dict, Any, Optional
import json

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.data_generator import SteelCastingDataGenerator
from utils.config_loader import ConfigLoader
from utils.logger import setup_project_logging
from utils.validation import DataQualityValidator

try:
    from tqdm import tqdm
except ImportError:
    print("Warning: tqdm not available. Progress bars will not be displayed.")
    tqdm = None


class ComprehensiveDataGenerator:
    """Comprehensive data generation with CLI interface, progress tracking, and validation"""
    
    def __init__(self, args: argparse.Namespace):
        """
        Initialize the comprehensive data generator.
        
        Args:
            args: Parsed command-line arguments
        """
        self.args = args
        self.start_time = time.time()
        
        # Setup logging
        log_level = "DEBUG" if args.verbose else "INFO"
        self.logger_instance = setup_project_logging("logs", log_level)
        self.logger = self.logger_instance.get_logger()
        
        # Load and merge configuration
        self.config_loader = ConfigLoader()
        self.config = self._load_and_merge_config()
        
        # Initialize data generator
        self.data_generator = SteelCastingDataGenerator(str(args.config))
        
        # Initialize validator if validation is requested
        self.validator = DataQualityValidator(self.config) if args.validate else None
        
        # Create output directories
        self.output_dir = Path(args.output_dir)
        self._create_output_directories()
        
        # Statistics tracking
        self.stats = {
            'total_casts': 0,
            'defect_count': 0,
            'validation_results': [],
            'generation_errors': [],
            'start_time': self.start_time
        }
        
        self.logger.info("Comprehensive Data Generator initialized")
        self.logger_instance.log_system_performance("initialization", operation="startup")
    
    def _load_and_merge_config(self) -> Dict[str, Any]:
        """Load configuration and apply CLI overrides."""
        # Load base configuration
        config = self.config_loader.load_yaml(Path(self.args.config).name)
        
        # Apply CLI overrides
        if hasattr(self.args, 'num_casts') and self.args.num_casts:
            config['data_generation']['num_casts'] = self.args.num_casts
            
        if hasattr(self.args, 'cast_duration') and self.args.cast_duration:
            config['data_generation']['cast_duration_minutes'] = self.args.cast_duration
            
        if hasattr(self.args, 'sampling_rate') and self.args.sampling_rate:
            config['data_generation']['sampling_rate_hz'] = 1.0 / self.args.sampling_rate
            
        if hasattr(self.args, 'defect_rate') and self.args.defect_rate is not None:
            config['data_generation']['defect_simulation']['defect_probability'] = self.args.defect_rate / 100.0
            
        if hasattr(self.args, 'noise_level') and self.args.noise_level is not None:
            # Scale noise std by noise level factor
            for sensor_name, sensor_config in config['data_generation']['sensors'].items():
                if isinstance(sensor_config, dict) and 'noise_std' in sensor_config:
                    sensor_config['noise_std'] *= self.args.noise_level
                    
        if hasattr(self.args, 'seed') and self.args.seed is not None:
            config['data_generation']['random_seed'] = self.args.seed
        
        return config
    
    def _create_output_directories(self) -> None:
        """Create organized output directory structure."""
        directories = [
            self.output_dir / 'raw_timeseries',
            self.output_dir / 'metadata',
            self.output_dir / 'summary',
            Path('logs')
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Output directories created under {self.output_dir}")
    
    def _save_generation_config(self) -> None:
        """Save the effective configuration used for generation."""
        config_path = self.output_dir / 'metadata' / 'generation_config.yaml'
        
        # Use the config_loader save method but with proper path handling
        with open(config_path, 'w') as f:
            import yaml
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        
        self.logger.info(f"Generation configuration saved to {config_path}")
    
    def _estimate_time_remaining(self, current_cast: int, total_casts: int) -> Optional[float]:
        """Estimate time remaining based on current progress."""
        if current_cast == 0:
            return None
            
        elapsed_time = time.time() - self.start_time
        rate = current_cast / elapsed_time
        remaining_casts = total_casts - current_cast
        
        return remaining_casts / rate if rate > 0 else None
    
    def generate_dataset(self) -> Dict[str, Any]:
        """
        Generate the complete dataset with progress tracking and validation.
        
        Returns:
            Dict containing generation summary and statistics
        """
        data_config = self.config['data_generation']
        num_casts = data_config['num_casts']
        
        self.logger_instance.log_data_generation_start(num_casts, self.config)
        
        # Save configuration
        self._save_generation_config()
        
        # Initialize progress tracking
        if tqdm:
            progress_bar = tqdm(
                total=num_casts,
                desc="Generating casts",
                unit="cast",
                disable=not sys.stdout.isatty()  # Disable in non-interactive environments
            )
        else:
            progress_bar = None
        
        all_metadata = []
        defect_count = 0
        
        try:
            # Generate all casts
            for i in range(num_casts):
                cast_id = f"cast_{i+1:04d}"
                
                try:
                    # Generate single cast
                    df, metadata = self.data_generator.generate_cast_sequence(cast_id)
                    
                    # Track statistics
                    if metadata['defect_label']:
                        defect_count += 1
                    
                    # Save time series data
                    output_file = self.output_dir / 'raw_timeseries' / f"{cast_id}.parquet"
                    df.to_parquet(output_file)
                    
                    # Store metadata
                    all_metadata.append(metadata)
                    
                    # Validate data if requested
                    if self.validator and (i % 100 == 0 or i < 10):  # Validate sample of casts
                        validation_results = self._validate_cast(df, metadata)
                        self.stats['validation_results'].extend(validation_results)
                    
                    # Update progress
                    current_cast = i + 1
                    eta = self._estimate_time_remaining(current_cast, num_casts)
                    
                    if progress_bar:
                        progress_bar.update(1)
                        if eta:
                            progress_bar.set_postfix({
                                'Defects': f"{defect_count}/{current_cast}",
                                'Rate': f"{defect_count/current_cast:.1%}",
                                'ETA': f"{eta/60:.1f}m"
                            })
                    
                    # Periodic logging
                    if current_cast % 100 == 0:
                        self.logger_instance.log_data_generation_progress(
                            current_cast, num_casts, defect_count, eta
                        )
                        self.logger_instance.log_system_performance(
                            "data_generator", operation=f"generated_{current_cast}_casts"
                        )
                
                except Exception as e:
                    error_msg = f"Error generating cast {cast_id}: {str(e)}"
                    self.logger.error(error_msg)
                    self.stats['generation_errors'].append({'cast_id': cast_id, 'error': str(e)})
                    
                    if not self.args.continue_on_error:
                        raise
            
            if progress_bar:
                progress_bar.close()
            
            # Save metadata and generate reports
            self._save_dataset_metadata(all_metadata, defect_count)
            self._generate_summary_reports(all_metadata, defect_count)
            
            # Final validation if requested
            if self.validator:
                self._perform_final_validation(all_metadata)
            
            # Update statistics
            self.stats.update({
                'total_casts': num_casts,
                'defect_count': defect_count,
                'defect_rate': defect_count / num_casts,
                'generation_time': time.time() - self.start_time
            })
            
            self.logger_instance.log_data_generation_end(
                num_casts, defect_count, self._get_quality_metrics()
            )
            
            return self.stats
            
        except KeyboardInterrupt:
            self.logger.warning("Generation interrupted by user")
            if progress_bar:
                progress_bar.close()
            
            # Save partial results
            if all_metadata:
                self._save_dataset_metadata(all_metadata, defect_count, partial=True)
            
            raise
        
        except Exception as e:
            self.logger.error(f"Generation failed: {str(e)}")
            if progress_bar:
                progress_bar.close()
            raise
    
    def _validate_cast(self, df, metadata) -> list:
        """Validate a single cast and return validation results."""
        validation_results = []
        
        try:
            # Statistical validation
            stat_result = self.validator.validate_statistical_properties(df, metadata['cast_id'])
            validation_results.append(stat_result)
            
            # Defect logic validation
            defect_result = self.validator.validate_defect_logic(df, metadata)
            validation_results.append(defect_result)
            
            # Temporal validation
            temporal_result = self.validator.validate_temporal_consistency(df, metadata['cast_id'])
            validation_results.append(temporal_result)
            
            # Log any validation issues
            for result in validation_results:
                if not result['passed']:
                    for issue in result['issues']:
                        self.logger_instance.log_data_quality_issue(
                            result['validation_type'], issue, "WARNING"
                        )
        
        except Exception as e:
            self.logger.error(f"Validation failed for {metadata['cast_id']}: {str(e)}")
        
        return validation_results
    
    def _perform_final_validation(self, all_metadata: list) -> None:
        """Perform final dataset-wide validation."""
        try:
            # Dataset distribution validation
            dist_result = self.validator.validate_dataset_distribution(all_metadata)
            self.stats['validation_results'].append(dist_result)
            
            # File integrity validation
            integrity_result = self.validator.validate_file_integrity(self.output_dir)
            self.stats['validation_results'].append(integrity_result)
            
            # Generate quality report
            quality_report_path = self.output_dir / 'summary' / 'quality_report.json'
            quality_report = self.validator.generate_quality_report(
                self.stats['validation_results'], quality_report_path
            )
            
            # Log validation summary
            self.logger_instance.log_validation_results(
                "final_validation", quality_report, quality_report['overall_passed']
            )
            
        except Exception as e:
            self.logger.error(f"Final validation failed: {str(e)}")
    
    def _save_dataset_metadata(self, all_metadata: list, defect_count: int, partial: bool = False) -> None:
        """Save comprehensive dataset metadata."""
        runtime = time.time() - self.start_time
        num_casts = len(all_metadata)
        
        dataset_metadata = {
            'dataset_info': {
                'total_casts': num_casts,
                'defect_count': defect_count,
                'defect_rate': defect_count / num_casts if num_casts > 0 else 0,
                'generation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'generation_time_seconds': runtime,
                'partial_generation': partial,
                'configuration': self.config,
                'cli_arguments': vars(self.args),
                'generation_errors': self.stats['generation_errors']
            },
            'cast_metadata': all_metadata
        }
        
        metadata_path = self.output_dir / 'metadata' / 'cast_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(dataset_metadata, f, indent=2)
        
        self.logger.info(f"Dataset metadata saved to {metadata_path}")
    
    def _generate_summary_reports(self, all_metadata: list, defect_count: int) -> None:
        """Generate summary reports and statistics."""
        # Dataset summary CSV
        summary_data = []
        for metadata in all_metadata:
            summary_data.append({
                'cast_id': metadata['cast_id'],
                'steel_grade': metadata['steel_grade'],
                'defect_label': metadata['defect_label'],
                'defect_triggers': ','.join(metadata['defect_trigger_events']),
                'avg_casting_speed': metadata['process_summary']['avg_casting_speed'],
                'avg_mold_temperature': metadata['process_summary']['avg_mold_temperature'],
                'avg_mold_level': metadata['process_summary']['avg_mold_level'],
                'generation_timestamp': metadata['generation_timestamp']
            })
        
        import pandas as pd
        summary_df = pd.DataFrame(summary_data)
        summary_path = self.output_dir / 'summary' / 'dataset_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        
        # Defect distribution report
        defect_distribution = {
            'total_casts': len(all_metadata),
            'defect_casts': defect_count,
            'normal_casts': len(all_metadata) - defect_count,
            'defect_rate': defect_count / len(all_metadata),
            'trigger_distribution': {},
            'grade_distribution': {}
        }
        
        # Count triggers and grades
        for metadata in all_metadata:
            # Count triggers
            for trigger in metadata['defect_trigger_events']:
                defect_distribution['trigger_distribution'][trigger] = \
                    defect_distribution['trigger_distribution'].get(trigger, 0) + 1
            
            # Count grades
            grade = metadata['steel_grade']
            defect_distribution['grade_distribution'][grade] = \
                defect_distribution['grade_distribution'].get(grade, 0) + 1
        
        defect_dist_path = self.output_dir / 'summary' / 'defect_distribution.json'
        with open(defect_dist_path, 'w') as f:
            json.dump(defect_distribution, f, indent=2)
        
        self.logger.info(f"Summary reports saved to {self.output_dir / 'summary'}")
    
    def _get_quality_metrics(self) -> Dict[str, Any]:
        """Get quality metrics from validation results."""
        if not self.stats['validation_results']:
            return {}
        
        metrics = {
            'total_validations': len(self.stats['validation_results']),
            'passed_validations': sum(1 for r in self.stats['validation_results'] if r.get('passed', False)),
            'validation_types': list(set(r.get('validation_type') for r in self.stats['validation_results']))
        }
        
        metrics['validation_pass_rate'] = metrics['passed_validations'] / metrics['total_validations']
        return metrics


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Comprehensive synthetic steel casting data generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python generate_synthetic_data.py
  
  # Generate 500 casts with custom defect rate
  python generate_synthetic_data.py --num-casts 500 --defect-rate 20
  
  # High-frequency sampling with validation
  python generate_synthetic_data.py --sampling-rate 0.5 --validate --verbose
  
  # Custom output directory with specific seed
  python generate_synthetic_data.py --output-dir data/experiment1 --seed 12345
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--config',
        type=str,
        default='configs/data_generation.yaml',
        help='Path to YAML configuration file (default: configs/data_generation.yaml)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/synthetic',
        help='Output directory for generated data (default: data/synthetic/)'
    )
    
    parser.add_argument(
        '--num-casts',
        type=int,
        help='Number of casting sequences to generate (overrides config)'
    )
    
    parser.add_argument(
        '--cast-duration',
        type=int,
        help='Duration of each cast in minutes (overrides config)'
    )
    
    parser.add_argument(
        '--sampling-rate',
        type=float,
        help='Data sampling frequency in seconds (overrides config)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--defect-rate',
        type=float,
        help='Percentage of casts with defects (0-100, overrides config)'
    )
    
    parser.add_argument(
        '--noise-level',
        type=float,
        default=1.0,
        help='Noise intensity factor (default: 1.0)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility (overrides config)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable detailed progress logging'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Run data quality validation after generation'
    )
    
    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help='Continue generation even if individual casts fail'
    )
    
    return parser


def main():
    """Main function to generate synthetic dataset with comprehensive CLI interface."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate arguments
    if args.defect_rate is not None and not (0 <= args.defect_rate <= 100):
        parser.error("Defect rate must be between 0 and 100")
    
    if args.noise_level is not None and args.noise_level < 0:
        parser.error("Noise level must be non-negative")
    
    if args.sampling_rate is not None and args.sampling_rate <= 0:
        parser.error("Sampling rate must be positive")
    
    # Check if config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found at {config_path}")
        print(f"Please ensure the configuration file exists or specify a different path with --config")
        sys.exit(1)
    
    try:
        print("Steel Casting Synthetic Data Generator")
        print("=" * 50)
        print(f"Configuration: {args.config}")
        print(f"Output directory: {args.output_dir}")
        if args.validate:
            print("Data validation: Enabled")
        print("-" * 50)
        
        # Initialize and run generator
        generator = ComprehensiveDataGenerator(args)
        stats = generator.generate_dataset()
        
        # Print summary
        print("\n" + "=" * 50)
        print("GENERATION COMPLETED SUCCESSFULLY")
        print("=" * 50)
        print(f"Total casts generated: {stats['total_casts']}")
        print(f"Defect casts: {stats['defect_count']}")
        print(f"Defect rate: {stats['defect_rate']:.2%}")
        print(f"Generation time: {stats['generation_time']:.1f} seconds")
        
        if stats['generation_errors']:
            print(f"Generation errors: {len(stats['generation_errors'])}")
        
        if args.validate and stats.get('validation_results'):
            quality_metrics = generator._get_quality_metrics()
            print(f"Validation pass rate: {quality_metrics.get('validation_pass_rate', 0):.2%}")
        
        print(f"Files saved to: {args.output_dir}")
        print(f"Logs saved to: logs/")
        
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during data generation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()