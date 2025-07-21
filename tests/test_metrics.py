import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.metrics import MetricsCalculator


class TestMetricsCalculator:
    """Test suite for MetricsCalculator"""

    def setup_method(self):
        """Setup test fixtures."""
        self.calculator = MetricsCalculator()

    def test_calculate_cost_sensitive_metrics_basic(self):
        """Test basic functionality of cost-sensitive metrics calculation."""
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([1, 1, 0, 0, 1])

        result = self.calculator.calculate_cost_sensitive_metrics(
            y_true, y_pred
        )

        # Check return type
        assert isinstance(result, dict)

        # Check required keys
        expected_keys = {
            "false_positive_cost",
            "false_negative_cost",
            "total_cost",
        }
        assert set(result.keys()) == expected_keys

        # Check values (1 FP with cost 1.0, 1 FN with cost 10.0)
        assert result["false_positive_cost"] == 1.0
        assert result["false_negative_cost"] == 10.0
        assert result["total_cost"] == 11.0

    def test_calculate_cost_sensitive_metrics_custom_costs(self):
        """Test cost-sensitive metrics with custom costs."""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 0])

        # 1 FP, 1 FN with custom costs
        result = self.calculator.calculate_cost_sensitive_metrics(
            y_true, y_pred, false_positive_cost=3.0, false_negative_cost=7.0
        )

        assert result["false_positive_cost"] == 3.0
        assert result["false_negative_cost"] == 7.0
        assert result["total_cost"] == 10.0

    def test_calculate_cost_sensitive_metrics_perfect_prediction(self):
        """Test cost-sensitive metrics with perfect predictions."""
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0, 1])

        result = self.calculator.calculate_cost_sensitive_metrics(
            y_true, y_pred
        )

        # No false positives or negatives
        assert result["false_positive_cost"] == 0.0
        assert result["false_negative_cost"] == 0.0
        assert result["total_cost"] == 0.0

    def test_calculate_cost_sensitive_metrics_all_wrong(self):
        """Test cost-sensitive metrics with completely wrong predictions."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0, 0, 1, 1])

        result = self.calculator.calculate_cost_sensitive_metrics(
            y_true, y_pred
        )

        # 2 FP with cost 1.0, 2 FN with cost 10.0
        assert result["false_positive_cost"] == 2.0
        assert result["false_negative_cost"] == 20.0
        assert result["total_cost"] == 22.0

    def test_calculate_cost_sensitive_metrics_empty_arrays(self):
        """Test cost-sensitive metrics with empty arrays."""
        y_true = np.array([])
        y_pred = np.array([])

        result = self.calculator.calculate_cost_sensitive_metrics(
            y_true, y_pred
        )

        # No predictions, no costs
        assert result["false_positive_cost"] == 0.0
        assert result["false_negative_cost"] == 0.0
        assert result["total_cost"] == 0.0

    def test_calculate_cost_sensitive_metrics_zero_costs(self):
        """Test cost-sensitive metrics with zero costs."""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 0])

        result = self.calculator.calculate_cost_sensitive_metrics(
            y_true, y_pred, false_positive_cost=0.0, false_negative_cost=0.0
        )

        assert result["false_positive_cost"] == 0.0
        assert result["false_negative_cost"] == 0.0
        assert result["total_cost"] == 0.0
