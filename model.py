import pickle
import torch
import torch.nn as nn
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

try:
    from nltk.sentiment import SentimentIntensityAnalyzer
except Exception:
    SentimentIntensityAnalyzer = None

try:
    from textblob import TextBlob
except Exception:
    TextBlob = None

# ==================================================================
# IMPORTANT: You need to have the class definitions from the original
# notebook (DistortionClassifier, EmotionClassifier, CustomJournalAnalyzer)
# copied into your Flask application's codebase for this to work.
# ==================================================================

class DistortionClassifier(nn.Module):
    """Neural network for cognitive distortion detection"""

    def __init__(self, input_dim, hidden_dim=256):
        super(DistortionClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x

class EmotionClassifier(nn.Module):
    """Neural network for emotion classification"""

    def __init__(self, input_dim, num_emotions, hidden_dim=256):
        super(EmotionClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, num_emotions)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.relu(self.fc3(x))
        return x

class CustomJournalAnalyzer:
    """Journal analyzer using custom trained models"""

    def __init__(self, distortion_model, emotion_model, vectorizer, emotion_encoder):
        self.distortion_model = distortion_model
        self.emotion_model = emotion_model
        self.vectorizer = vectorizer
        self.emotion_encoder = emotion_encoder
        self._vader = self._init_vader()

        self.distortion_patterns = {
            'all_or_nothing': [
                r'\b(always|never|every|no one|everyone|everything|nothing|completely|totally)\b'
            ],
            'overgeneralization': [
                r'\b(always happens|never works|typical|as usual|every time)\b'
            ],
            'catastrophizing': [
                r'\b(disaster|catastrophe|terrible|horrible|awful|worst|ruined|destroyed)\b'
            ],
            'personalization': [
                r'\b(my fault|I\'m to blame|because of me|I caused)\b'
            ],
            'should_statements': [
                r'\b(should|shouldn\'t|must|mustn\'t|ought to|have to)\b'
            ],
            'labeling': [
                r'\b(i\'m a|i am a)\b.*\b(loser|failure|idiot|stupid)\b'
            ]
        }

    def _init_vader(self):
        """Initialize VADER if available; otherwise return None."""
        if SentimentIntensityAnalyzer is None:
            return None
        try:
            return SentimentIntensityAnalyzer()
        except Exception:
            return None

    def _compute_vader_sentiment(self, text):
        if self._vader is not None:
            try:
                scores = self._vader.polarity_scores(text)
                return {
                    'pos': float(scores.get('pos', 0.0)),
                    'neu': float(scores.get('neu', 1.0)),
                    'neg': float(scores.get('neg', 0.0)),
                    'compound': float(scores.get('compound', 0.0)),
                }
            except Exception:
                pass
        return {'pos': 0.0, 'neu': 1.0, 'neg': 0.0, 'compound': 0.0}

    def _compute_textblob_sentiment(self, text):
        if TextBlob is not None:
            try:
                sentiment = TextBlob(text).sentiment
                return {
                    'polarity': float(getattr(sentiment, 'polarity', 0.0)),
                    'subjectivity': float(getattr(sentiment, 'subjectivity', 0.0)),
                }
            except Exception:
                pass
        return {'polarity': 0.0, 'subjectivity': 0.0}

    def _overall_sentiment_label(self, vader_sentiment):
        compound = float(vader_sentiment.get('compound', 0.0))
        if compound >= 0.05:
            return 'positive'
        if compound <= -0.05:
            return 'negative'
        return 'neutral'

    def detect_distortions_pattern(self, text):
        """Detect specific distortion patterns"""
        text_lower = text.lower()
        detected = {}

        for distortion, patterns in self.distortion_patterns.items():
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, text_lower, flags=re.IGNORECASE)
                if found:
                    matches.extend(found)

            if matches:
                detected[distortion] = {
                    'found': True,
                    'count': len(matches),
                    'examples': matches[:3]
                }

        return detected

    def analyze_entry(self, text):
        """Analyze a journal entry"""
        # Vectorize text
        X = self.vectorizer.transform([text]).toarray()
        X_tensor = torch.FloatTensor(X)

        # Predict distortion
        self.distortion_model.eval()
        with torch.no_grad():
            distortion_prob = self.distortion_model(X_tensor).item()

        is_distorted = distortion_prob > 0.5

        # Predict emotion
        self.emotion_model.eval()
        with torch.no_grad():
            emotion_outputs = self.emotion_model(X_tensor)
            emotion_probs = torch.softmax(emotion_outputs, dim=1).squeeze().numpy()
            emotion_idx = torch.argmax(emotion_outputs, dim=1).item()

        primary_emotion = self.emotion_encoder.classes_[emotion_idx]

        # Get emotion distribution
        emotion_scores = {
            emotion: float(prob)
            for emotion, prob in zip(self.emotion_encoder.classes_, emotion_probs)
        }

        # Detect specific distortion patterns
        distortions = self.detect_distortions_pattern(text)

        # Sentiment analysis for frontend-friendly values
        vader_sentiment = self._compute_vader_sentiment(text)
        textblob_sentiment = self._compute_textblob_sentiment(text)
        overall_sentiment = self._overall_sentiment_label(vader_sentiment)

        # Assessment
        health_assessment = self._assess_mental_health(
            distortion_prob, primary_emotion, distortions
        )

        return {
            'text': text,
            'is_distorted': is_distorted,
            'distortion_score': distortion_prob,
            'distortions_detected': distortions,
            'emotion_analysis': {
                'emotions': emotion_scores,
                'primary_emotion': primary_emotion,
                'primary_emotion_score': emotion_scores[primary_emotion],
                'vader_sentiment': vader_sentiment,
                'textblob_sentiment': textblob_sentiment,
                'overall_sentiment': overall_sentiment,
            },
            'health_assessment': health_assessment
        }

    def _assess_mental_health(self, distortion_score, primary_emotion, distortions):
        """Assess mental health based on analysis"""
        concerns = []

        if distortion_score > 0.7:
            concerns.append("High level of cognitive distortions detected")
        elif distortion_score > 0.5:
            concerns.append("Moderate cognitive distortions present")

        if primary_emotion in ['sadness', 'fear', 'anger'] and distortion_score > 0.6:
            concerns.append(f"Strong {primary_emotion} with distorted thinking")

        if len(distortions) >= 3:
            concerns.append("Multiple distortion patterns identified")

        if concerns:
            level = 'high' if len(concerns) >= 2 else 'moderate'
        else:
            level = 'low'

        recommendations = {
            'low': "Your thinking patterns appear healthy. Keep journaling!",
            'moderate': "Some distortions detected. Try challenging these thoughts.",
            'high': "Multiple concerning patterns. Consider professional support."
        }

        return {
            'concern_level': level,
            'concerns': concerns,
            'recommendation': recommendations[level]
        }

    def visualize_analysis(self, analysis):
        """Visualize analysis results"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Custom Model Journal Analysis', fontsize=16, fontweight='bold')

        # 1. Distortion Score
        ax1 = axes[0, 0]
        score = analysis['distortion_score']
        colors = ['green' if score < 0.3 else 'orange' if score < 0.7 else 'red']
        ax1.barh(['Distortion'], [score], color=colors)
        ax1.set_xlim([0, 1])
        ax1.set_title('Distortion Probability')
        ax1.axvline(x=0.5, color='black', linestyle='--', alpha=0.5)

        # 2. Emotion Distribution
        ax2 = axes[0, 1]
        emotions = analysis['emotion_analysis']['emotions']
        em_names = list(emotions.keys())
        em_scores = list(emotions.values())
        colors_em = plt.cm.Set3(range(len(em_names)))
        ax2.pie(em_scores, labels=em_names, autopct='%1.1f%%', colors=colors_em)
        ax2.set_title('Emotion Distribution')

        # 3. Specific Distortions
        ax3 = axes[1, 0]
        if analysis['distortions_detected']:
            dist_names = [d.replace('_', ' ').title()
                         for d in analysis['distortions_detected'].keys()]
            dist_counts = [analysis['distortions_detected'][d]['count']
                          for d in analysis['distortions_detected'].keys()]
            ax3.barh(dist_names, dist_counts, color='coral')
            ax3.set_xlabel('Count')
            ax3.set_title('Specific Distortions Found')
        else:
            ax3.text(0.5, 0.5, 'No Specific\nDistortions Detected',
                    ha='center', va='center', fontsize=12)
            ax3.axis('off')

        # 4. Summary
        ax4 = axes[1, 1]
        ax4.axis('off')

        summary = f"""
        ANALYSIS SUMMARY

        Status: {'⚠ DISTORTED' if analysis['is_distorted'] else '✓ HEALTHY'}

        Distortion Score: {analysis['distortion_score']:.2%}

        Primary Emotion: {analysis['emotion_analysis']['primary_emotion'].title()}
        ({analysis['emotion_analysis']['primary_emotion_score']:.1%})

        Concern Level: {analysis['health_assessment']['concern_level'].upper()}

        Recommendation:
        {analysis['health_assessment']['recommendation']}
        """

        ax4.text(0.1, 0.5, summary, fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()
        plt.show()

    def print_report(self, analysis):
        """Print detailed report"""
        print("\n" + "="*80)
        print("JOURNAL ANALYSIS REPORT (Custom Models)")
        print("="*80)

        print(f"\n📝 Entry: {analysis['text'][:150]}...")
        print(f"\n{'⚠ DISTORTED' if analysis['is_distorted'] else '✓ HEALTHY'}")
        print(f"Distortion Probability: {analysis['distortion_score']:.2%}")

        if analysis['distortions_detected']:
            print(f"\n🧠 Specific Distortions:")
            for dist, details in analysis['distortions_detected'].items():
                print(f"  • {dist.replace('_', ' ').title()}: {details['count']} occurrences")

        print(f"\n😊 Emotions:")
        print(f"  Primary: {analysis['emotion_analysis']['primary_emotion'].title()} "
              f"({analysis['emotion_analysis']['primary_emotion_score']:.1%})")

        print(f"\n💡 Assessment:")
        print(f"  Level: {analysis['health_assessment']['concern_level'].upper()}")
        print(f"  {analysis['health_assessment']['recommendation']}")
        print("\n" + "="*80)

# --- Loading the models and components ---

filepath = 'journal_analyzer_models.pkl'

# Global variables to store loaded models
_loaded_distortion_model = None
_loaded_emotion_model = None
_loaded_vectorizer = None
_loaded_emotion_encoder = None
_analyzer_instance = None

def load_models():
    """Load the ML models from pickle file"""
    global _loaded_distortion_model, _loaded_emotion_model, _loaded_vectorizer, _loaded_emotion_encoder
    
    try:
        with open(filepath, 'rb') as f:
            loaded_data = pickle.load(f)

        # Load vectorizer and emotion encoder directly
        _loaded_vectorizer = loaded_data['vectorizer']
        _loaded_emotion_encoder = loaded_data['emotion_encoder']

        # Reconstruct Distortion Model
        # You need the input_dim from the vectorizer AFTER it's loaded
        input_dim = _loaded_vectorizer.shape_[1] if hasattr(_loaded_vectorizer, 'shape_') else _loaded_vectorizer.transform(['test']).shape[1]
        _loaded_distortion_model = DistortionClassifier(input_dim)
        _loaded_distortion_model.load_state_dict(loaded_data['distortion_model'])
        _loaded_distortion_model.eval() # Set to evaluation mode

        # Reconstruct Emotion Model
        num_emotions = len(_loaded_emotion_encoder.classes_)
        _loaded_emotion_model = EmotionClassifier(input_dim, num_emotions)
        _loaded_emotion_model.load_state_dict(loaded_data['emotion_model'])
        _loaded_emotion_model.eval() # Set to evaluation mode

        print("✅ Models and components loaded successfully!")
        print(f"Vectorizer vocabulary size: {len(_loaded_vectorizer.vocabulary_)}")
        print(f"Emotion classes: {_loaded_emotion_encoder.classes_}")

        return True

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found. Please ensure it's in the correct directory.")
        return False
    except Exception as e:
        print(f"An error occurred while loading the models: {e}")
        return False

def get_analyzer():
    """Get the analyzer instance, loading models if necessary"""
    global _analyzer_instance
    
    if _analyzer_instance is None:
        # Check if models are loaded
        if (_loaded_distortion_model is None or _loaded_emotion_model is None or 
            _loaded_vectorizer is None or _loaded_emotion_encoder is None):
            # Try to load models
            if not load_models():
                raise RuntimeError("Failed to load ML models")
        
        # Create analyzer instance
        _analyzer_instance = CustomJournalAnalyzer(
            _loaded_distortion_model,
            _loaded_emotion_model,
            _loaded_vectorizer,
            _loaded_emotion_encoder
        )
    
    return _analyzer_instance

# Try to load models on import (optional - will fail silently if file not found)
try:
    load_models()
except:
    pass  # Models will be loaded on first get_analyzer() call
