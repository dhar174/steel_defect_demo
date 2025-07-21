#!/usr/bin/env python3
"""
Test script for feature analysis notebook functionality
"""

import sys
import os
sys.path.append('../src')

# Test imports
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.figure_factory as ff

    # Statistical analysis
    from scipy import stats
    from scipy.stats import chi2_contingency, pearsonr, spearmanr
    from sklearn.feature_selection import (
        mutual_info_classif, SelectKBest, f_classif,
        chi2, RFE
    )
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import cross_val_score

    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")
    exit(1)

# Test custom modules
try:
    from data.data_loader import DataLoader
    from features.feature_engineer import CastingFeatureEngineer
    print("✓ Custom modules imported successfully")
except Exception as e:
    print(f"✗ Custom module import error: {e}")
    exit(1)

# Test data loading
try:
    data_loader = DataLoader(data_dir='../data')
    df = data_loader.load_cleaned_data('../data/processed/cleaned_data.csv')
    print(f"✓ Data loaded successfully: {df.shape}")
    print(f"✓ Target distribution: {df['defect'].value_counts().to_dict()}")
except Exception as e:
    print(f"✗ Data loading error: {e}")
    exit(1)

# Test basic analysis functions
try:
    # Separate features and target
    X = df.drop('defect', axis=1)
    y = df['defect']
    
    print(f"✓ Features separated: {X.shape[1]} features, {len(y)} samples")
    
    # Test correlation analysis
    numeric_df = df.select_dtypes(include=[np.number])
    pearson_corr = numeric_df.corr(method='pearson')
    print(f"✓ Correlation analysis completed: {pearson_corr.shape}")
    
    # Test feature importance
    rf = RandomForestClassifier(n_estimators=10, random_state=42)  # Reduced for speed
    rf.fit(X, y)
    print(f"✓ Random Forest training completed")
    
    # Test statistical tests
    normal_data = df[df['defect'] == 0].iloc[:, 0]  # First feature
    defect_data = df[df['defect'] == 1].iloc[:, 0]
    stat_test = stats.mannwhitneyu(normal_data.dropna(), defect_data.dropna(), alternative='two-sided')
    print(f"✓ Statistical tests completed")
    
    print("\n🎉 All tests passed! The notebook should work correctly.")
    
except Exception as e:
    print(f"✗ Analysis error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)