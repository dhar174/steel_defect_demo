# Defect Labeling Validation Guide for Steel Casting

## Overview

This guide explains the defect labeling validation system implemented for the steel casting quality monitoring system. The validation framework ensures that synthetic defect labels align with domain knowledge and identifies edge cases requiring expert review.

## Validation Components

### 1. Label Distribution Analysis

**Purpose**: Examine the balance and distribution of defect vs. good classifications across the dataset.

**Key Metrics**:
- Overall defect rate vs. target rate
- Class balance ratio (Good:Defect)
- Trigger frequency and effectiveness
- Distribution by steel grade
- Statistical comparisons between defect and good casts

**Domain Significance**: Ensures the synthetic data reflects realistic defect rates observed in actual steel casting operations.

### 2. Domain Knowledge Validation

**Purpose**: Validate that defect conditions align with established steel casting principles and domain expertise.

**Validation Rules**:

#### Mold Level Deviation
- **Normal Range**: 130-170mm (configurable)
- **Critical Threshold**: >20 seconds outside normal range
- **Domain Logic**: Extended mold level deviations cause shell formation issues and potential breakouts
- **Severity Assessment**: Based on duration and magnitude of deviation

#### Rapid Temperature Drop
- **Threshold**: >50°C drop in 60 seconds
- **Severe Threshold**: >75°C drop
- **Domain Logic**: Rapid cooling causes thermal stress and potential shell cracking
- **Assessment**: Considers drop magnitude and cooling rate

#### High Speed with Low Superheat
- **Critical Combination**: Speed >1.5 m/min with superheat <20°C
- **Very Risky**: Speed >1.7 m/min with superheat <18°C
- **Domain Logic**: Insufficient superheat at high casting speeds increases breakout risk
- **Assessment**: Evaluates duration of risky operating conditions

### 3. Edge Case Detection

**Purpose**: Identify borderline cases that may be mislabeled or require expert judgment.

**Detection Criteria**:

#### Borderline Conditions
- Near-threshold trigger conditions (within 5mm, 10°C, or 0.1 m/min of limits)
- Operating parameters at boundaries of normal ranges
- Inconsistent signal patterns

#### Conflicting Signals
- High superheat with excessive cooling (contradictory)
- Low casting speed with high temperature (inefficient)
- Stable mold level with high flow variations (unusual)

#### Labeling Inconsistencies
- Defect labels without identifiable triggers
- Trigger conditions present but labeled as good
- All parameters in normal ranges but labeled as defect

### 4. Expert Review Documentation

**Purpose**: Provide comprehensive documentation for domain expert validation and decision-making.

**Report Components**:

#### Executive Summary
- Dataset overview and key statistics
- Major findings and concerns
- Overall assessment rating

#### Detailed Findings
- Comprehensive analysis results
- Trigger effectiveness metrics
- Statistical comparisons

#### Recommendations
- Specific actions to improve labeling accuracy
- Threshold adjustments suggestions
- Additional validation criteria needs

#### Appendices
- Failed validation cases
- High uncertainty cases
- Detailed statistical analysis
- Methodology documentation

## Validation Workflow

### 1. Data Generation
```bash
python scripts/validate_defect_labeling.py --sample-size 100 --generate-data
```

### 2. Analysis Execution
The validation performs:
1. Label distribution analysis across entire dataset
2. Domain knowledge validation on sample of casts
3. Edge case detection and uncertainty scoring
4. Expert review documentation generation

### 3. Results Interpretation

#### Pass Rates
- **Excellent (90-100%)**: Defect labeling highly consistent with domain knowledge
- **Good (80-89%)**: Minor issues, generally sound labeling
- **Fair (70-79%)**: Several issues, recommend review and refinement  
- **Poor (<70%)**: Significant issues, major review required

#### Edge Case Rates
- **Low (<10%)**: Well-defined labeling criteria
- **Moderate (10-30%)**: Some borderline cases expected
- **High (>30%)**: May indicate unclear criteria or threshold issues

### 4. Action Items Based on Results

#### High Defect Rate Deviation
- Review and adjust base defect probability
- Examine trigger threshold settings
- Consider environmental factors

#### Low Trigger Coverage
- Add new trigger conditions
- Review existing trigger logic
- Consider multi-factor trigger combinations

#### High Edge Case Rate
- Refine threshold definitions
- Add intermediate severity categories
- Develop additional validation criteria

## Domain Expert Input Requirements

### Critical Review Areas

1. **Trigger Condition Validity**
   - Do the trigger thresholds reflect real-world critical conditions?
   - Are there missing trigger types that should be included?
   - Are severity assessments aligned with operational experience?

2. **Edge Case Assessment**
   - Review high-uncertainty cases for labeling accuracy
   - Validate borderline condition classifications
   - Assess conflicting signal interpretations

3. **Operational Alignment**
   - Do defect rates match plant experience?
   - Are grade-specific variations realistic?
   - Do temporal patterns reflect actual operations?

### Feedback Integration

The validation system is designed to incorporate expert feedback through:
- Threshold adjustments based on domain knowledge
- Additional trigger condition development
- Severity assessment refinement
- Edge case reclassification guidelines

## Technical Implementation

### Configuration
Key validation parameters are configurable in the domain rules:
```python
domain_rules = {
    'mold_level_critical_deviation': 20,  # seconds
    'temperature_drop_severe': 75,        # °C
    'speed_superheat_critical': (1.7, 18), # (m/min, °C)
    'normal_operation_ranges': { ... }
}
```

### Extensibility
The validation framework supports:
- Addition of new trigger types
- Custom domain rule implementation
- Integration with real plant data
- Automated validation workflows

### Quality Metrics
The system provides quantitative measures for:
- Labeling consistency
- Domain alignment
- Edge case identification
- Expert review prioritization

## Usage Examples

### Basic Validation
```bash
python scripts/validate_defect_labeling.py --sample-size 50
```

### Custom Configuration
```bash
python scripts/validate_defect_labeling.py --config custom_config.yaml --sample-size 100
```

### Detailed Analysis
```bash
python scripts/validate_defect_labeling.py --sample-size 200 --verbose --output-dir detailed_results
```

## Conclusion

This validation framework provides a systematic approach to ensuring defect labeling quality in synthetic steel casting data. By combining automated analysis with domain expert review, it helps maintain alignment between synthetic conditions and real-world steel casting knowledge.

The multi-layered validation approach catches both systematic labeling issues and edge cases, providing confidence in the quality of training data for machine learning models while highlighting areas that require expert attention.