"""
Explainability module for model interpretation.

Provides SHAP-based analysis for understanding trading model decisions.
"""

from .shap_analyzer import ShapAnalyzer

__all__ = ['ShapAnalyzer']
