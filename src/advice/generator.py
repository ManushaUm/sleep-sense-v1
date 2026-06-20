import numpy as np
from src.advice.llm_generator import TransformerAdviceGenerator

# Instantiate the LLM-based Sleep Coach
llm_coach = TransformerAdviceGenerator()
try:
    # Attempt to load Hugging Face pipeline (will fall back to CoT engine if offline or missing cache)
    llm_coach.initialize_pipeline()
except Exception:
    pass

def generate_advice(top_shap_features: list[dict]) -> list[str]:
    """
    Generate actionable sleep advice based on the top 3 SHAP features and their values
    using a systematic, few-shot prompt-engineered Transformer (or CoT fallback).
    
    Args:
        top_shap_features: A list of dicts:
            [{'feature': str, 'shap_value': float, 'feature_value': float}]
            
    Returns:
        List of 3 advice strings.
    """
    return llm_coach.generate_recommendations(top_shap_features)
