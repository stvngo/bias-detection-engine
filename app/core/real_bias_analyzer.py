"""
Real bias analysis using NLP techniques and established bias detection methods
"""
import re
import json
import time
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Tuple
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import os
import structlog

logger = structlog.get_logger()

class RealBiasAnalyzer:
    """Real bias detection using established NLP techniques"""
    
    def __init__(self):
        """Initialize the real bias analyzer"""
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
            
        # Expanded bias-indicating word lists (based on research)
        self.loaded_words = {
            'positive': ['amazing', 'breakthrough', 'revolutionary', 'historic', 'unprecedented', 
                        'triumphant', 'victory', 'success', 'achievement', 'progress', 'excellent',
                        'outstanding', 'remarkable', 'exceptional', 'impressive', 'wonderful'],
            'negative': ['devastating', 'catastrophic', 'disaster', 'failure', 'crisis', 
                        'collapse', 'chaos', 'scandal', 'corruption', 'betrayal', 'terrible',
                        'awful', 'shocking', 'outrageous', 'disturbing', 'alarming'],
            'partisan_left': ['progressive', 'reform', 'justice', 'equality', 'inclusive', 
                             'sustainable', 'grassroots', 'community', 'diversity', 'social justice',
                             'climate action', 'systemic', 'marginalized', 'oppressed'],
            'partisan_right': ['traditional', 'conservative', 'freedom', 'liberty', 'patriotic', 
                              'security', 'law and order', 'values', 'family values', 'free market',
                              'individual responsibility', 'constitutional', 'fiscal'],
            'hedging': ['allegedly', 'reportedly', 'supposedly', 'claims', 'sources say', 
                       'it appears', 'seems to', 'may have', 'could be', 'might be', 'possible',
                       'appears to', 'suggests', 'indicates', 'believed to'],
            'certainty': ['definitely', 'certainly', 'undoubtedly', 'clearly', 'obviously', 
                         'without question', 'proves', 'confirms', 'established',
                         'undeniable', 'absolute', 'guaranteed', 'conclusive']
        }
        
        # Emotional intensity words
        self.emotion_amplifiers = ['very', 'extremely', 'incredibly', 'absolutely', 'totally', 
                                  'completely', 'utterly', 'massively', 'severely', 'deeply']
    
    def _calculate_ideological_stance(self, text: str) -> Tuple[int, List[Dict]]:
        """Calculate ideological bias based on partisan language patterns"""
        text_lower = text.lower()
        
        left_score = sum(1 for word in self.loaded_words['partisan_left'] if word in text_lower)
        right_score = sum(1 for word in self.loaded_words['partisan_right'] if word in text_lower)
        
        # Find actual phrases that contributed to the score
        phrases = []
        for word in self.loaded_words['partisan_left']:
            if word in text_lower:
                phrases.append({
                    'text': word,
                    'bias_direction': 'left',
                    'score_impact': {'ideological_stance': -5}
                })
        
        for word in self.loaded_words['partisan_right']:
            if word in text_lower:
                phrases.append({
                    'text': word,
                    'bias_direction': 'right', 
                    'score_impact': {'ideological_stance': 5}
                })
        
        # Calculate score (50 = neutral, 0 = left, 100 = right)
        total_partisan = left_score + right_score
        if total_partisan == 0:
            score = 50  # Neutral
        else:
            bias_ratio = right_score / total_partisan
            score = int(bias_ratio * 100)
        
        return score, phrases
    
    def _calculate_factual_grounding(self, text: str) -> Tuple[int, List[Dict]]:
        """Assess factual grounding based on hedging vs certainty language"""
        text_lower = text.lower()
        
        hedging_count = sum(1 for phrase in self.loaded_words['hedging'] if phrase in text_lower)
        certainty_count = sum(1 for phrase in self.loaded_words['certainty'] if phrase in text_lower)
        
        phrases = []
        for phrase in self.loaded_words['hedging']:
            if phrase in text_lower:
                phrases.append({
                    'text': phrase,
                    'type': 'hedging',
                    'score_impact': {'factual_grounding': 10}
                })
        
        for phrase in self.loaded_words['certainty']:
            if phrase in text_lower:
                phrases.append({
                    'text': phrase,
                    'type': 'certainty',
                    'score_impact': {'factual_grounding': -5}
                })
        
        # Higher hedging = more factual grounding (careful language)
        # Lower hedging = less factual grounding (overconfident claims)
        if hedging_count + certainty_count == 0:
            score = 60  # Default moderate
        else:
            hedging_ratio = hedging_count / (hedging_count + certainty_count)
            score = int(30 + (hedging_ratio * 50))  # 30-80 range
        
        return score, phrases
    
    def _calculate_framing_choices(self, text: str) -> Tuple[int, List[Dict]]:
        """Analyze framing through loaded language detection"""
        text_lower = text.lower()
        
        positive_loaded = sum(1 for word in self.loaded_words['positive'] if word in text_lower)
        negative_loaded = sum(1 for word in self.loaded_words['negative'] if word in text_lower)
        
        phrases = []
        for word in self.loaded_words['positive']:
            if word in text_lower:
                phrases.append({
                    'text': word,
                    'type': 'positive_loaded',
                    'score_impact': {'framing_choices': -10}
                })
        
        for word in self.loaded_words['negative']:
            if word in text_lower:
                phrases.append({
                    'text': word,
                    'type': 'negative_loaded',
                    'score_impact': {'framing_choices': -10}
                })
        
        # More loaded language = lower framing score (more biased)
        total_loaded = positive_loaded + negative_loaded
        word_count = len(text.split())
        
        if word_count == 0:
            score = 50
        else:
            loaded_ratio = total_loaded / word_count * 100
            score = max(20, int(80 - (loaded_ratio * 30)))  # 20-80 range
        
        return score, phrases
    
    def _calculate_emotional_tone(self, text: str) -> Tuple[int, List[Dict]]:
        """Calculate emotional bias using VADER sentiment + amplifier detection"""
        sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
        
        # Count emotion amplifiers
        text_lower = text.lower()
        amplifier_count = sum(1 for amp in self.emotion_amplifiers if amp in text_lower)
        
        phrases = []
        for amp in self.emotion_amplifiers:
            if amp in text_lower:
                phrases.append({
                    'text': amp,
                    'type': 'emotion_amplifier',
                    'score_impact': {'emotional_tone': -8}
                })
        
        # Calculate emotional neutrality score (higher = more neutral)
        compound_abs = abs(sentiment_scores['compound'])
        amplifier_penalty = min(20, amplifier_count * 5)
        
        score = int(80 - (compound_abs * 40) - amplifier_penalty)
        score = max(10, min(90, score))  # 10-90 range
        
        return score, phrases
    
    def _calculate_source_transparency(self, text: str, source: str) -> Tuple[int, List[Dict]]:
        """Assess source transparency based on attribution patterns"""
        text_lower = text.lower()
        
        # Look for source attribution patterns
        attribution_patterns = [
            r'according to \w+',
            r'sources? (say|said|tell|told)',
            r'officials? (say|said|tell|told)',
            r'experts? (say|said|believe|think)',
            r'study (shows|finds|reveals)',
            r'data (shows|indicates|suggests)'
        ]
        
        attribution_count = 0
        phrases = []
        
        for pattern in attribution_patterns:
            matches = re.findall(pattern, text_lower)
            attribution_count += len(matches)
            for match in matches:
                phrases.append({
                    'text': match,
                    'type': 'source_attribution',
                    'score_impact': {'source_transparency': 5}
                })
        
        # Anonymous source patterns (reduce transparency)
        anonymous_patterns = [
            r'anonymous sources?',
            r'unnamed officials?',
            r'sources? close to',
            r'insiders? (say|said)',
            r'sources? familiar with'
        ]
        
        anonymous_count = 0
        for pattern in anonymous_patterns:
            matches = re.findall(pattern, text_lower)
            anonymous_count += len(matches)
            for match in matches:
                phrases.append({
                    'text': match,
                    'type': 'anonymous_source',
                    'score_impact': {'source_transparency': -8}
                })
        
        # Calculate transparency score
        word_count = len(text.split())
        if word_count == 0:
            score = 50
        else:
            attribution_ratio = attribution_count / word_count * 100
            anonymous_penalty = anonymous_count * 10
            score = int(40 + (attribution_ratio * 30) - anonymous_penalty)
            score = max(20, min(85, score))  # 20-85 range
        
        return score, phrases
    
    def _calculate_confidence_intervals(self, scores: Dict[str, int], phrases_count: int) -> Dict[str, List[int]]:
        """Calculate real confidence intervals based on analysis robustness"""
        intervals = {}
        
        for dimension, score in scores.items():
            # Confidence based on amount of evidence found
            if phrases_count >= 5:
                margin = 8  # High confidence
            elif phrases_count >= 3:
                margin = 12  # Medium confidence
            else:
                margin = 18  # Low confidence
            
            intervals[dimension] = [
                max(0, score - margin),
                min(100, score + margin)
            ]
        
        return intervals
    
    async def analyze_article(self, title: str, content: str, source: str) -> Dict[str, Any]:
        """Perform real bias analysis using NLP techniques"""
        start_time = time.time()
        
        # Combine title and content for analysis
        full_text = f"{title}. {content}"
        
        logger.info("Starting real bias analysis", 
                   text_length=len(full_text), source=source)
        
        # Calculate each dimension with real metrics
        ideological_score, ideological_phrases = self._calculate_ideological_stance(full_text)
        factual_score, factual_phrases = self._calculate_factual_grounding(full_text)
        framing_score, framing_phrases = self._calculate_framing_choices(full_text)
        emotional_score, emotional_phrases = self._calculate_emotional_tone(full_text)
        transparency_score, transparency_phrases = self._calculate_source_transparency(full_text, source)
        
        # Combine all highlighted phrases
        all_phrases = (ideological_phrases + factual_phrases + framing_phrases + 
                      emotional_phrases + transparency_phrases)
        
        # If no specific bias phrases found, extract general meaningful phrases
        if len(all_phrases) == 0:
            all_phrases = self._extract_fallback_phrases(full_text)
        
        # Add explanations to phrases
        for phrase in all_phrases:
            if 'explanation' not in phrase:
                phrase['explanation'] = self._generate_phrase_explanation(phrase)
                phrase['confidence'] = phrase.get('confidence', 0.7)  # Default confidence
        
        dimension_scores = {
            "ideological_stance": ideological_score,
            "factual_grounding": factual_score, 
            "framing_choices": framing_score,
            "emotional_tone": emotional_score,
            "source_transparency": transparency_score
        }
        
        confidence_intervals = self._calculate_confidence_intervals(dimension_scores, len(all_phrases))
        overall_score = sum(dimension_scores.values()) / len(dimension_scores)
        processing_time_ms = (time.time() - start_time) * 1000
        
        logger.info("Real bias analysis completed",
                   processing_time_ms=processing_time_ms,
                   phrases_found=len(all_phrases),
                   scores=dimension_scores)
        
        return {
            "article_id": f"{source}_{hash(title) % 10000}",
            "overall_score": round(overall_score, 1),
            "dimension_scores": dimension_scores,
            "confidence_intervals": confidence_intervals,
            "highlighted_phrases": all_phrases[:10],  # Limit to top 10
            "processing_time_ms": round(processing_time_ms, 2),
            "model_used": "NLP_RealAnalysis_v1.0",
            "timestamp": datetime.now().isoformat(),
            "analysis_method": "rule_based_nlp_with_sentiment"
        }
    
    def _extract_fallback_phrases(self, text: str) -> List[Dict]:
        """Extract meaningful phrases when no specific bias indicators are found"""
        sentences = text.split('.')[:3]  # Take first 3 sentences
        fallback_phrases = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Only meaningful sentences
                # Look for key phrases like quotes, assertions, or claims
                if any(word in sentence.lower() for word in ['said', 'according to', 'reported', 'claims', 'states']):
                    # Extract a key phrase (up to 60 characters)
                    phrase_text = sentence[:60] + '...' if len(sentence) > 60 else sentence
                    fallback_phrases.append({
                        'text': phrase_text,
                        'type': 'neutral_statement',
                        'score_impact': {'overall': 0},
                        'confidence': 0.5,
                        'explanation': f"Key statement from article: '{phrase_text}'"
                    })
        
        # If still no phrases, use title and source
        if len(fallback_phrases) == 0:
            fallback_phrases.append({
                'text': text.split('.')[0][:50] + '...' if len(text.split('.')[0]) > 50 else text.split('.')[0],
                'type': 'headline',
                'score_impact': {'overall': 0},
                'confidence': 0.3,
                'explanation': "Article headline for reference"
            })
        
        return fallback_phrases[:3]  # Max 3 fallback phrases

    def _generate_phrase_explanation(self, phrase: Dict) -> str:
        """Generate explanations for detected bias phrases"""
        phrase_type = phrase.get('type', phrase.get('bias_direction', 'unknown'))
        
        explanations = {
            'left': f"'{phrase['text']}' indicates progressive/liberal ideological framing",
            'right': f"'{phrase['text']}' indicates conservative/traditional ideological framing", 
            'hedging': f"'{phrase['text']}' shows careful, qualified language increasing factual reliability",
            'certainty': f"'{phrase['text']}' shows overconfident language that may indicate bias",
            'positive_loaded': f"'{phrase['text']}' is emotionally positive loaded language",
            'negative_loaded': f"'{phrase['text']}' is emotionally negative loaded language",
            'emotion_amplifier': f"'{phrase['text']}' amplifies emotional intensity beyond neutral reporting",
            'source_attribution': f"'{phrase['text']}' provides proper source attribution",
            'anonymous_source': f"'{phrase['text']}' relies on anonymous sources, reducing transparency",
            'neutral_statement': f"'{phrase['text']}' is a key factual statement from the article",
            'headline': f"'{phrase['text']}' is the article headline or key opening statement"
        }
        
        return explanations.get(phrase_type, f"'{phrase['text']}' may indicate reporting bias")
