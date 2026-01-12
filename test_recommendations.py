import pytest
from ..services.hr_recommend import RecommendSystem
from ..services.simpleLLM import SimpleOllamaEngine

def test_recommend_system():
    rs = RecommendSystem()
    recommendations = rs.get_recommendations(['OverTime', 'MonthlyIncome'])
    assert len(recommendations) <= 3
    assert all(isinstance(rec, str) for rec in recommendations)

def test_ollama_availability():
    engine = SimpleOllamaEngine()
    is_available = engine.is_available()
    assert isinstance(is_available, bool)