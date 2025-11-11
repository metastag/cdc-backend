
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import warnings
warnings.filterwarnings('ignore')

class CognitiveDistortionDetector:
    """Detects cognitive distortions in text using pattern matching and NLP"""

    def __init__(self):
        self.distortion_patterns = {
            'all_or_nothing': [
                r'\b(always|never|every|no one|everyone|everything|nothing|completely|totally|absolute)\b',
                r'\b(all the time|not a single)\b'
            ],
            'overgeneralization': [
                r'\b(always happens|never works|typical|of course|as usual)\b',
                r'\b(every time|each time)\b'
            ],
            'mental_filter': [
                r'\b(only|just|merely|simply)\b.*\b(bad|negative|wrong|terrible)\b',
                r'\b(can\'t see|don\'t see|ignore)\b.*\b(good|positive)\b'
            ],
            'disqualifying_positive': [
                r'\b(but|however|although|even though)\b.*\b(doesn\'t count|doesn\'t matter|not important)\b',
                r'\b(yeah but|yes but|okay but)\b',
                r'\b(just luck|just because|anyone could)\b'
            ],
            'jumping_to_conclusions': [
                r'\b(must be|has to be|obviously|clearly|definitely)\b.*\b(thinks|feels|believes)\b',
                r'\b(I know|I\'m sure).*\b(thinks|hates|dislikes)\b'
            ],
            'catastrophizing': [
                r'\b(disaster|catastrophe|terrible|horrible|awful|worst)\b',
                r'\b(ruined|destroyed|devastated|doomed)\b',
                r'\b(end of the world|can\'t handle|unbearable)\b'
            ],
            'emotional_reasoning': [
                r'\b(I feel).*\b(therefore|so it must|means)\b',
                r'\b(feels like).*\b(must be|is)\b'
            ],
            'should_statements': [
                r'\b(should|shouldn\'t|must|mustn\'t|ought to|have to|need to)\b',
                r'\b(supposed to)\b'
            ],
            'labeling': [
                r'\b(I\'m a|I am a|I\'m an|I am an)\b.*\b(loser|failure|idiot|stupid|worthless)\b',
                r'\b(he\'s a|she\'s a|they\'re)\b.*\b(jerk|idiot|fool)\b'
            ],
            'personalization': [
                r'\b(my fault|I\'m to blame|because of me|I caused)\b',
                r'\b(I should have|if only I|I didn\'t)\b'
            ]
        }

    def detect_distortions(self, text):
        """Detect cognitive distortions in text"""
        text_lower = text.lower()
        detected = {}

        for distortion, patterns in self.distortion_patterns.items():
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, text_lower)
                if found:
                    matches.extend(found)

            if matches:
                detected[distortion] = {
                    'found': True,
                    'count': len(matches),
                    'examples': matches[:3]
                }

        return detected

    def get_distortion_score(self, text):
        """Calculate overall distortion score (0-1)"""
        detections = self.detect_distortions(text)
        if not detections:
            return 0.0

        total_score = sum(d['count'] for d in detections.values())
        # Normalize by text length and cap at 1.0
        words = len(text.split())
        normalized_score = min(total_score / max(words * 0.1, 1), 1.0)
        return normalized_score

class EmotionAnalyzer:
    """Analyzes emotional tone using transformer models and sentiment analysis"""

    def __init__(self):
        print("Loading emotion detection model...")
        self.emotion_classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None
        )
        self.vader = SentimentIntensityAnalyzer()
        print("Emotion analyzer ready!")

    def analyze_emotions(self, text):
        """Analyze emotions in text"""
        # Get emotion predictions
        emotions = self.emotion_classifier(text[:512])[0]  # Limit text length

        # Get VADER sentiment
        vader_scores = self.vader.polarity_scores(text)

        # Get TextBlob sentiment
        blob = TextBlob(text)
        textblob_sentiment = {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }

        # Format emotion results
        emotion_scores = {e['label']: e['score'] for e in emotions}
        primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])

        return {
            'emotions': emotion_scores,
            'primary_emotion': primary_emotion[0],
            'primary_emotion_score': primary_emotion[1],
            'vader_sentiment': vader_scores,
            'textblob_sentiment': textblob_sentiment,
            'overall_sentiment': self._get_overall_sentiment(vader_scores)
        }

    def _get_overall_sentiment(self, vader_scores):
        """Determine overall sentiment from VADER scores"""
        compound = vader_scores['compound']
        if compound >= 0.05:
            return 'positive'
        elif compound <= -0.05:
            return 'negative'
        else:
            return 'neutral'

class JournalAnalyzer:
    """Main class that combines distortion detection and emotion analysis"""

    def __init__(self):
        print("Initializing Journal Analyzer...")
        self.distortion_detector = CognitiveDistortionDetector()
        self.emotion_analyzer = EmotionAnalyzer()
        print("Journal Analyzer ready!")

    def analyze_entry(self, text):
        """Perform complete analysis of a journal entry"""
        # Detect cognitive distortions
        distortions = self.distortion_detector.detect_distortions(text)
        distortion_score = self.distortion_detector.get_distortion_score(text)

        # Analyze emotions
        emotion_analysis = self.emotion_analyzer.analyze_emotions(text)

        # Determine if entry is distorted
        is_distorted = distortion_score > 0.15

        return {
            'text': text,
            'is_distorted': is_distorted,
            'distortion_score': distortion_score,
            'distortions_detected': distortions,
            'emotion_analysis': emotion_analysis,
            'health_assessment': self._assess_mental_health(distortion_score, emotion_analysis)
        }

    def _assess_mental_health(self, distortion_score, emotion_analysis):
        """Provide mental health assessment"""
        sentiment = emotion_analysis['overall_sentiment']
        primary_emotion = emotion_analysis['primary_emotion']

        concerns = []

        if distortion_score > 0.3:
            concerns.append("High level of cognitive distortions detected")
        elif distortion_score > 0.15:
            concerns.append("Moderate cognitive distortions present")

        if sentiment == 'negative' and distortion_score > 0.15:
            concerns.append("Negative sentiment combined with distorted thinking")

        if primary_emotion in ['sadness', 'fear', 'anger'] and distortion_score > 0.2:
            concerns.append(f"Strong {primary_emotion} with distorted thinking patterns")

        if concerns:
            level = 'high' if len(concerns) >= 2 else 'moderate'
        else:
            level = 'low'

        return {
            'concern_level': level,
            'concerns': concerns,
            'recommendation': self._get_recommendation(level)
        }

    def _get_recommendation(self, level):
        """Get recommendation based on concern level"""
        recommendations = {
            'low': "Your journal entry shows healthy thinking patterns. Continue journaling!",
            'moderate': "Some cognitive distortions detected. Try to challenge these thoughts and consider alternative perspectives.",
            'high': "Multiple cognitive distortions and concerning patterns detected. Consider speaking with a mental health professional for support."
        }
        return recommendations.get(level, recommendations['low'])

    def visualize_analysis(self, analysis):
        """Create visualization of the analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Journal Entry Analysis', fontsize=16, fontweight='bold')

        # 1. Cognitive Distortions
        ax1 = axes[0, 0]
        if analysis['distortions_detected']:
            distortions = analysis['distortions_detected']
            names = [d.replace('_', ' ').title() for d in distortions.keys()]
            counts = [distortions[d]['count'] for d in distortions.keys()]

            ax1.barh(names, counts, color='coral')
            ax1.set_xlabel('Frequency')
            ax1.set_title('Cognitive Distortions Detected')
            ax1.invert_yaxis()
        else:
            ax1.text(0.5, 0.5, 'No Distortions Detected',
                    ha='center', va='center', fontsize=12)
            ax1.set_title('Cognitive Distortions Detected')
        ax1.axis('off') if not analysis['distortions_detected'] else None

        # 2. Emotion Distribution
        ax2 = axes[0, 1]
        emotions = analysis['emotion_analysis']['emotions']
        em_names = list(emotions.keys())
        em_scores = list(emotions.values())

        colors = plt.cm.Set3(range(len(em_names)))
        ax2.pie(em_scores, labels=em_names, autopct='%1.1f%%', colors=colors)
        ax2.set_title('Emotion Distribution')

        # 3. Sentiment Scores
        ax3 = axes[1, 0]
        vader = analysis['emotion_analysis']['vader_sentiment']
        sentiment_types = ['Positive', 'Neutral', 'Negative']
        sentiment_scores = [vader['pos'], vader['neu'], vader['neg']]

        bars = ax3.bar(sentiment_types, sentiment_scores,
                      color=['lightgreen', 'lightblue', 'lightcoral'])
        ax3.set_ylabel('Score')
        ax3.set_title('Sentiment Analysis (VADER)')
        ax3.set_ylim([0, 1])

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')

        # 4. Overall Health Assessment
        ax4 = axes[1, 1]
        ax4.axis('off')

        # Create summary text
        summary = f"""
        ANALYSIS SUMMARY

        Status: {'⚠️ DISTORTED' if analysis['is_distorted'] else '✓ HEALTHY'}

        Distortion Score: {analysis['distortion_score']:.2%}

        Primary Emotion: {analysis['emotion_analysis']['primary_emotion'].title()}

        Overall Sentiment: {analysis['emotion_analysis']['overall_sentiment'].title()}

        Concern Level: {analysis['health_assessment']['concern_level'].upper()}

        Recommendation:
        {analysis['health_assessment']['recommendation']}
        """

        ax4.text(0.1, 0.5, summary, fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()
        plt.show()

    def print_detailed_report(self, analysis):
        """Print detailed analysis report"""
        print("\n" + "="*80)
        print("JOURNAL ENTRY ANALYSIS REPORT")
        print("="*80)

        print(f"\n📝 JOURNAL ENTRY:")
        print(f"{analysis['text'][:200]}..." if len(analysis['text']) > 200 else analysis['text'])

        print(f"\n{'⚠️ DISTORTED THINKING DETECTED' if analysis['is_distorted'] else '✓ HEALTHY THINKING PATTERNS'}")
        print(f"Distortion Score: {analysis['distortion_score']:.2%}")

        if analysis['distortions_detected']:
            print(f"\n🧠 COGNITIVE DISTORTIONS FOUND:")
            for distortion, details in analysis['distortions_detected'].items():
                print(f"\n  • {distortion.replace('_', ' ').title()}:")
                print(f"    - Frequency: {details['count']}")
                print(f"    - Examples: {', '.join(str(e) for e in details['examples'][:2])}")

        print(f"\n😊 EMOTIONAL ANALYSIS:")
        emotions = analysis['emotion_analysis']['emotions']
        print(f"  Primary Emotion: {analysis['emotion_analysis']['primary_emotion'].title()} "
              f"({analysis['emotion_analysis']['primary_emotion_score']:.1%})")
        print(f"  Overall Sentiment: {analysis['emotion_analysis']['overall_sentiment'].title()}")
        print(f"\n  Emotion Breakdown:")
        for emotion, score in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
            print(f"    - {emotion.title()}: {score:.1%}")

        print(f"\n💡 MENTAL HEALTH ASSESSMENT:")
        health = analysis['health_assessment']
        print(f"  Concern Level: {health['concern_level'].upper()}")
        if health['concerns']:
            print(f"  Concerns:")
            for concern in health['concerns']:
                print(f"    • {concern}")
        print(f"\n  Recommendation:")
        print(f"  {health['recommendation']}")

        print("\n" + "="*80 + "\n")

# Factory function for Flask integration
def create_analyzer():
    """Create and return a JournalAnalyzer instance for Flask backend"""
    return JournalAnalyzer()

# Optional: Create a singleton instance for reuse
_analyzer_instance = None

def get_analyzer():
    """Get singleton analyzer instance to avoid reloading models"""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = create_analyzer()
    return _analyzer_instance