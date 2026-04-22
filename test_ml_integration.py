#!/usr/bin/env python3
"""
Test script to demonstrate the ML model integration with the Flask backend.
"""

from services.analysis import compute_analysis_for_text
from model import get_analyzer

def test_ml_integration():
    """Test the complete ML integration"""
    print("="*60)
    print("Testing ML Model Integration with Flask Backend")
    print("="*60)
    
    # Test cases
    test_cases = [
        {
            "name": "High distortion case",
            "text": "I messed up on that presentation. I always do this, I'm such a failure. Everyone is going to think I'm incompetent."
        },
        {
            "name": "Moderate distortion case", 
            "text": "I should have done better on that test. I need to study more next time."
        },
        {
            "name": "Healthy thinking case",
            "text": "I'm feeling grateful for my supportive friends. I'm learning new skills and making progress."
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['name']}:")
        print(f"Text: \"{case['text']}\"")
        print("-" * 40)
        
        # Run analysis through services layer (used by Flask app)
        result = compute_analysis_for_text(case['text'])
        
        print(f"Overall Score: {result['overallScore']}/100")
        print(f"Rule-based Distortions: {len(result['distortions'])}")
        print(f"Positive Patterns: {len(result['positivePatterns'])}")
        
        if 'model' in result:
            model_result = result['model']
            print(f"ML Distortion Score: {model_result['distortion_score']:.2%}")
            print(f"ML Primary Emotion: {model_result['emotion_analysis']['primary_emotion']}")
            print(f"ML Concern Level: {model_result['health_assessment']['concern_level']}")
        
        print()

if __name__ == "__main__":
    test_ml_integration()