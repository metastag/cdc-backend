#!/usr/bin/env python3
"""Quick test of the cleaned model.py"""

from model import get_analyzer

def test_analyzer():
    print("Loading analyzer...")
    analyzer = get_analyzer()
    
    test_text = "I always mess everything up and nothing ever works"
    print(f"Testing with: '{test_text}'")
    
    result = analyzer.analyze_entry(test_text)
    
    print("✓ Analysis completed successfully!")
    print(f"Distortion score: {result['distortion_score']:.2f}")
    print(f"Is distorted: {result['is_distorted']}")
    print(f"Primary emotion: {result['emotion_analysis']['primary_emotion']}")
    print(f"Overall sentiment: {result['emotion_analysis']['overall_sentiment']}")
    
    # Test with a positive text
    positive_text = "I had a great day and everything went well"
    result2 = analyzer.analyze_entry(positive_text)
    print(f"\nPositive test - Distortion score: {result2['distortion_score']:.2f}")
    print(f"Primary emotion: {result2['emotion_analysis']['primary_emotion']}")

if __name__ == "__main__":
    test_analyzer()