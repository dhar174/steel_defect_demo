"""
Example: Using the Comprehensive Model Evaluation Framework
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import yaml

# Import the evaluation framework
from src.models.model_evaluator import ModelEvaluator
from src.models.baseline_model import BaselineXGBoostModel
from src.models.evaluation_metrics import CustomMetrics
from src.models.evaluation_reports import EvaluationReports

def main():
    """Example usage of the comprehensive evaluation framework"""
    
    print("🔧 Steel Defect Detection - Model Evaluation Example")
    print("=" * 60)
    
    # 1. Generate example data
    print("\n1. Generating synthetic steel defect data...")
    X, y = make_classification(
        n_samples=800, n_features=15, n_classes=2,
        weights=[0.8, 0.2], random_state=42
    )
    
    feature_names = [
        'temperature', 'pressure', 'vibration', 'thickness',
        'surface_roughness', 'hardness', 'chemical_carbon',
        'chemical_silicon', 'cooling_rate', 'heating_rate',
        'humidity', 'oxygen_level', 'electromagnetic_reading',
        'ultrasonic_reading', 'visual_score'
    ]
    
    X_df = pd.DataFrame(X, columns=feature_names)
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"✓ Created dataset: {len(X)} samples, {len(feature_names)} features")
    print(f"✓ Defect rate: {y.mean():.1%}")
    
    # 2. Train model
    print("\n2. Training XGBoost model...")
    model = BaselineXGBoostModel(random_state=42)
    model.fit(X_train, y_train)
    print("✓ Model trained successfully")
    
    # 3. Initialize evaluator
    print("\n3. Setting up comprehensive evaluator...")
    evaluator = ModelEvaluator(
        model=model,
        model_name="SteelDefect_Example",
        output_dir="results/example_evaluation",
        save_plots=True
    )
    print("✓ Evaluator initialized")
    
    # 4. Perform evaluation
    print("\n4. Running comprehensive evaluation...")
    results = evaluator.evaluate_model(X_test, y_test)
    
    # Display results
    metrics = results['metrics']
    print(f"\n📊 EVALUATION RESULTS:")
    print(f"   ROC AUC: {metrics['roc_auc']:.3f}")
    print(f"   Precision: {metrics['precision']:.3f}")
    print(f"   Recall: {metrics['recall']:.3f}")
    print(f"   F1 Score: {metrics['f1_score']:.3f}")
    
    # 5. Calculate steel-specific metrics
    print("\n5. Calculating steel-specific metrics...")
    y_pred = model.predict(X_test)
    steel_metrics = CustomMetrics.manufacturing_kpi_suite(y_test, y_pred)
    
    print(f"   Defect Detection Rate: {steel_metrics['defect_detection_rate']:.3f}")
    print(f"   False Alarm Rate: {steel_metrics['false_alarm_rate']:.3f}")
    print(f"   Production Impact Score: ${steel_metrics['production_impact_score']:.0f}")
    
    # 6. Generate visualizations
    print("\n6. Generating visualizations...")
    y_proba = model.predict_proba(X_test)
    
    # ROC and PR curves
    roc_fig = evaluator.plot_roc_curve(y_test, y_proba)
    pr_fig = evaluator.plot_precision_recall_curve(y_test, y_proba)
    cm_fig = evaluator.plot_confusion_matrix(y_test, y_pred)
    
    print("✓ Key plots generated")
    
    # 7. Cross-validation analysis
    print("\n7. Running cross-validation...")
    cv_results = evaluator.cross_validation_analysis(
        model.model, X_train, y_train, cv_folds=5, scoring=['roc_auc', 'f1']
    )
    
    for metric, result in cv_results['metrics'].items():
        print(f"   CV {metric}: {result['test_mean']:.3f} ± {result['test_std']:.3f}")
    
    # 8. Find optimal threshold
    print("\n8. Finding optimal threshold...")
    optimal_thresh, optimal_f1 = evaluator.find_optimal_threshold(y_test, y_proba, 'f1')
    print(f"   Optimal F1 threshold: {optimal_thresh:.3f} (F1: {optimal_f1:.3f})")
    
    # 9. Generate report
    print("\n9. Generating evaluation report...")
    report_generator = EvaluationReports(output_dir="results/example_reports")
    
    model_info = {
        'model_type': 'XGBoost',
        'n_estimators': model.model_params.get('n_estimators', 100),
        'training_samples': len(X_train),
        'features': len(feature_names)
    }
    
    report_paths = report_generator.generate_comprehensive_report(
        results, model_info, include_plots=True
    )
    
    print("✓ Reports generated:")
    for format_name, path in report_paths.items():
        print(f"   {format_name}: {path}")
    
    print(f"\n✅ Evaluation completed successfully!")
    print(f"🎯 Model performance: ROC AUC = {metrics['roc_auc']:.3f}")
    print(f"🏭 Production ready: {'Yes' if metrics['roc_auc'] > 0.8 else 'Needs improvement'}")
    
    return results

if __name__ == "__main__":
    results = main()