#!/usr/bin/env python3
"""
Comprehensive defect labeling validation script for steel casting data.

This script performs detailed validation of defect labeling logic, including:
1. Label distribution analysis
2. Domain knowledge validation
3. Edge case detection
4. Expert review documentation generation

Usage:
    python scripts/validate_defect_labeling.py [--config CONFIG_PATH] [--data-dir DATA_DIR] [--output-dir OUTPUT_DIR]
"""

import argparse
import sys
from pathlib import Path
import json
import pandas as pd
from datetime import datetime
import logging

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from utils.config_loader import ConfigLoader
from utils.defect_labeling_validator import DefectLabelingValidator
from data.data_generator import SteelCastingDataGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Validate defect labeling in steel casting data'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/data_generation.yaml',
        help='Path to configuration file (default: configs/data_generation.yaml)'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Directory containing generated data (default: data)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='validation_results',
        help='Directory to save validation results (default: validation_results)'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=100,
        help='Number of casts to generate for validation (default: 100)'
    )
    
    parser.add_argument(
        '--generate-data',
        action='store_true',
        help='Generate new data for validation (default: use existing data)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def load_or_generate_data(config_path: str, data_dir: str, sample_size: int, generate_new: bool):
    """Load existing data or generate new data for validation."""
    data_dir_path = Path(data_dir)
    
    if generate_new or not (data_dir_path / 'synthetic').exists():
        logger.info(f"Generating {sample_size} casts for validation...")
        
        # Create temporary config for validation
        with open(config_path, 'r') as f:
            import yaml
            config = yaml.safe_load(f)
        
        # Override settings for validation
        config['data_generation']['num_casts'] = sample_size
        config['data_generation']['progress_reporting_frequency'] = max(1, sample_size // 10)
        
        # Save temporary config
        temp_config_path = Path('temp_validation_config.yaml')
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Generate data
        generator = SteelCastingDataGenerator(str(temp_config_path))
        generator.generate_dataset()
        
        # Cleanup
        temp_config_path.unlink()
        
        logger.info("Data generation completed")
    
    # Load metadata
    metadata_path = data_dir_path / 'synthetic' / 'dataset_metadata.json'
    if not metadata_path.exists():
        raise FileNotFoundError(f"Dataset metadata not found at {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        dataset_info = json.load(f)
    
    return dataset_info


def load_cast_data(data_dir: str, cast_metadata: dict) -> pd.DataFrame:
    """Load time series data for a specific cast."""
    cast_id = cast_metadata['cast_id']
    
    # Find the corresponding parquet file
    data_dir_path = Path(data_dir)
    parquet_files = list((data_dir_path / 'raw').glob('*.parquet'))
    
    # Match by cast number (extract from cast_id)
    cast_number = int(cast_id.split('_')[-1])
    expected_file = data_dir_path / 'raw' / f'cast_timeseries_{cast_number:04d}.parquet'
    
    if expected_file.exists():
        return pd.read_parquet(expected_file)
    
    # Fallback: try to find any matching file
    for parquet_file in parquet_files:
        if f'{cast_number:04d}' in parquet_file.name:
            return pd.read_parquet(parquet_file)
    
    raise FileNotFoundError(f"Time series data not found for cast {cast_id}")


def run_validation_analysis(validator: DefectLabelingValidator, 
                           dataset_info: dict, 
                           data_dir: str,
                           output_dir: str):
    """Run comprehensive validation analysis."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    cast_metadata_list = dataset_info['cast_metadata']
    
    logger.info("Starting defect labeling validation analysis...")
    
    # 1. Label Distribution Analysis
    logger.info("Performing label distribution analysis...")
    label_distribution = validator.analyze_label_distribution(cast_metadata_list)
    
    # Save label distribution results
    label_distribution_converted = validator._convert_for_json(label_distribution)
    with open(output_path / 'label_distribution_analysis.json', 'w') as f:
        json.dump(label_distribution_converted, f, indent=2)
    
    logger.info(f"Label distribution analysis completed. Defect rate: {label_distribution['dataset_summary']['defect_rate']:.1%}")
    
    # 2. Domain Knowledge Validation
    logger.info("Performing domain knowledge validation...")
    domain_validations = []
    
    # Sample a subset for detailed analysis if dataset is large
    sample_size = min(50, len(cast_metadata_list))  # Limit to 50 for performance
    sample_metadata = cast_metadata_list[:sample_size]
    
    for i, cast_metadata in enumerate(sample_metadata):
        try:
            # Load cast time series data
            df = load_cast_data(data_dir, cast_metadata)
            
            # Validate domain knowledge
            domain_result = validator.validate_domain_knowledge_alignment(df, cast_metadata)
            domain_validations.append(domain_result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Completed domain validation for {i + 1}/{len(sample_metadata)} casts")
                
        except Exception as e:
            logger.warning(f"Failed to validate cast {cast_metadata['cast_id']}: {e}")
            continue
    
    # Save domain validation results
    domain_validations_converted = validator._convert_for_json(domain_validations)
    with open(output_path / 'domain_knowledge_validation.json', 'w') as f:
        json.dump(domain_validations_converted, f, indent=2)
    
    passed_validations = len([dv for dv in domain_validations if dv.get('passed', True)])
    logger.info(f"Domain knowledge validation completed. Pass rate: {passed_validations}/{len(domain_validations)} ({passed_validations/len(domain_validations):.1%})")
    
    # 3. Edge Case Detection
    logger.info("Performing edge case detection...")
    edge_cases = []
    
    for i, cast_metadata in enumerate(sample_metadata):
        try:
            # Load cast time series data
            df = load_cast_data(data_dir, cast_metadata)
            
            # Detect edge cases
            edge_case_result = validator.identify_edge_cases(df, cast_metadata)
            edge_cases.append(edge_case_result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Completed edge case detection for {i + 1}/{len(sample_metadata)} casts")
                
        except Exception as e:
            logger.warning(f"Failed to analyze edge cases for cast {cast_metadata['cast_id']}: {e}")
            continue
    
    # Save edge case results
    edge_cases_converted = validator._convert_for_json(edge_cases)
    with open(output_path / 'edge_case_analysis.json', 'w') as f:
        json.dump(edge_cases_converted, f, indent=2)
    
    review_cases = len([ec for ec in edge_cases if ec.get('requires_expert_review', False)])
    logger.info(f"Edge case detection completed. {review_cases}/{len(edge_cases)} cases require expert review")
    
    # 4. Generate Expert Review Documentation
    logger.info("Generating expert review documentation...")
    expert_review = validator.generate_expert_review_documentation(
        label_distribution=label_distribution,
        domain_validations=domain_validations,
        edge_cases=edge_cases,
        output_path=output_path / 'expert_review_report.json'
    )
    
    # Generate summary report
    generate_summary_report(
        label_distribution, domain_validations, edge_cases, expert_review, output_path
    )
    
    logger.info(f"Validation analysis completed. Results saved to {output_path}")
    
    return {
        'label_distribution': label_distribution,
        'domain_validations': domain_validations,
        'edge_cases': edge_cases,
        'expert_review': expert_review
    }


def generate_summary_report(label_distribution: dict, 
                           domain_validations: list, 
                           edge_cases: list,
                           expert_review: dict, 
                           output_path: Path):
    """Generate a human-readable summary report."""
    
    summary_lines = [
        "# Steel Casting Defect Labeling Validation Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Executive Summary",
        ""
    ]
    
    # Dataset overview
    dataset_summary = label_distribution['dataset_summary']
    summary_lines.extend([
        f"**Total Casts Analyzed:** {dataset_summary['total_casts']}",
        f"**Defect Rate:** {dataset_summary['defect_rate']:.1%} (Target: {dataset_summary['target_defect_rate']:.1%})",
        f"**Class Balance Ratio:** {dataset_summary['class_balance_ratio']:.1f}:1 (Good:Defect)",
        ""
    ])
    
    # Key findings
    key_findings = expert_review['executive_summary']['key_findings']
    if key_findings:
        summary_lines.extend([
            "## Key Findings",
            ""
        ])
        for finding in key_findings:
            summary_lines.append(f"- {finding}")
        summary_lines.append("")
    
    # Domain validation summary
    passed_validations = len([dv for dv in domain_validations if dv.get('passed', True)])
    total_validations = len(domain_validations)
    
    summary_lines.extend([
        "## Domain Knowledge Validation",
        f"**Pass Rate:** {passed_validations}/{total_validations} ({passed_validations/total_validations:.1%})",
        ""
    ])
    
    # Edge case summary
    review_cases = len([ec for ec in edge_cases if ec.get('requires_expert_review', False)])
    total_cases = len(edge_cases)
    
    summary_lines.extend([
        "## Edge Case Analysis",
        f"**Cases Requiring Expert Review:** {review_cases}/{total_cases} ({review_cases/total_cases:.1%})",
        ""
    ])
    
    # Trigger analysis
    trigger_analysis = label_distribution.get('trigger_analysis', {})
    individual_triggers = trigger_analysis.get('individual_triggers', {})
    
    if individual_triggers:
        summary_lines.extend([
            "## Trigger Distribution",
            ""
        ])
        for trigger, count in individual_triggers.items():
            summary_lines.append(f"- **{trigger.replace('_', ' ').title()}:** {count} occurrences")
        summary_lines.append("")
    
    # Overall assessment
    overall_assessment = expert_review['executive_summary']['overall_assessment']
    summary_lines.extend([
        "## Overall Assessment",
        f"{overall_assessment}",
        ""
    ])
    
    # Recommendations
    recommendations = expert_review.get('recommendations', [])
    if recommendations:
        summary_lines.extend([
            "## Recommendations",
            ""
        ])
        for rec in recommendations:
            summary_lines.append(f"- {rec}")
        summary_lines.append("")
    
    # Save summary report
    summary_content = "\n".join(summary_lines)
    with open(output_path / 'validation_summary.md', 'w') as f:
        f.write(summary_content)
    
    # Also save as plain text for easy reading
    with open(output_path / 'validation_summary.txt', 'w') as f:
        f.write(summary_content)


def main():
    """Main execution function."""
    args = parse_arguments()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config_loader = ConfigLoader()
        config = config_loader.load_yaml(Path(args.config).name)
        
        # Load or generate data
        logger.info("Loading/generating validation data...")
        dataset_info = load_or_generate_data(
            args.config, args.data_dir, args.sample_size, args.generate_data
        )
        
        # Initialize validator
        logger.info("Initializing defect labeling validator...")
        validator = DefectLabelingValidator(config)
        
        # Run validation analysis
        results = run_validation_analysis(
            validator, dataset_info, args.data_dir, args.output_dir
        )
        
        # Print summary to console
        print("\n" + "="*60)
        print("DEFECT LABELING VALIDATION SUMMARY")
        print("="*60)
        
        dataset_summary = results['label_distribution']['dataset_summary']
        print(f"Total Casts: {dataset_summary['total_casts']}")
        print(f"Defect Rate: {dataset_summary['defect_rate']:.1%}")
        
        passed_validations = len([dv for dv in results['domain_validations'] if dv.get('passed', True)])
        total_validations = len(results['domain_validations'])
        print(f"Domain Validation Pass Rate: {passed_validations}/{total_validations} ({passed_validations/total_validations:.1%})")
        
        review_cases = len([ec for ec in results['edge_cases'] if ec.get('requires_expert_review', False)])
        total_cases = len(results['edge_cases'])
        print(f"Edge Cases Requiring Review: {review_cases}/{total_cases} ({review_cases/total_cases:.1%})")
        
        overall_assessment = results['expert_review']['executive_summary']['overall_assessment']
        print(f"\nOverall Assessment: {overall_assessment}")
        
        print(f"\nDetailed results saved to: {args.output_dir}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()