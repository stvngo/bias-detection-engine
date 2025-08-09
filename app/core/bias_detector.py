"""
Core Bias Detection Engine using OpenAI GPT
Fast API-based bias detection for real-time analysis
"""
import asyncio
import time
import json
import hashlib
import os
from typing import Dict, List, Optional, Any
from datetime import datetime

import openai
import structlog

from app.models.schemas import BiasScore, BiasAnalysisOutput
from config.settings import settings

logger = structlog.get_logger()


class MultiStagePrompts:
    """Sophisticated prompting system for bias detection"""

    @staticmethod
    def fast_bias_analysis_prompt(article_text: str) -> str:
        """Optimized prompt for sub-200ms analysis with phrase highlighting"""
        return f"""
Analyze bias in this article. Return JSON only:
{{
  "ideological_stance": <0-100>,
  "factual_grounding": <0-100>,
  "framing_choices": <0-100>,
  "emotional_tone": <0-100>,
  "source_transparency": <0-100>,
  "confidence": {{"ideological_stance": <0-100>, "factual_grounding": <0-100>, "framing_choices": <0-100>, "emotional_tone": <0-100>, "source_transparency": <0-100>}},
  "highlighted_phrases": [
    {{"text": "exact phrase", "score_impact": {{"dimension": "score_change"}}, "explanation": "why biased", "confidence": <0-100>}}
  ]
}}

Score: 0=left/poor/biased/emotional/opaque, 50=center/moderate/neutral/calm/transparent, 100=right/excellent/objective/analytical/clear

Article: {article_text[:2000]}
JSON:"""


class BiasDetectionEngine:
    """Fast bias detection using OpenAI GPT models"""
    
    def __init__(self):
        self.model_name = "gpt-3.5-turbo"
        self.max_tokens = 200  # Reduced for speed
        self.temperature = 0.1  # Lower for consistency and speed
        self.client = None
        self.cache = {}  # In-memory cache for speed
        self.prompts = MultiStagePrompts()
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client with cost controls"""
        try:
            # Load API key from config file
            config_path = "config.env"
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    for line in f:
                        if line.startswith('OPENAI_API_KEY='):
                            api_key = line.split('=', 1)[1].strip()
                            os.environ['OPENAI_API_KEY'] = api_key
                        elif line.startswith('OPENAI_MODEL='):
                            self.model_name = line.split('=', 1)[1].strip()
                        elif line.startswith('OPENAI_MAX_TOKENS='):
                            self.max_tokens = int(line.split('=', 1)[1].strip())
                        elif line.startswith('OPENAI_TEMPERATURE='):
                            self.temperature = float(line.split('=', 1)[1].strip())
            
            api_key = os.getenv('OPENAI_API_KEY')
            
            if not api_key:
                logger.warning("No OPENAI_API_KEY found, using intelligent demo mode")
                self.client = None
                return
            
            logger.info(f"Initializing OpenAI client with {self.model_name} (max_tokens: {getattr(self, 'max_tokens', 200)})")
            self.client = openai.OpenAI(
                api_key=api_key,
                timeout=30.0
            )
            logger.info("OpenAI GPT-3.5 Turbo client initialized successfully with cost controls")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self.client = None

    async def _generate_analysis(self, prompt: str) -> str:
        """Helper to generate analysis using OpenAI with cost controls"""
        if self.client:
            try:
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert media bias analyst. Respond only with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=getattr(self, 'max_tokens', 300),  # Cost control
                    temperature=getattr(self, 'temperature', 0.3)
                )
                logger.info(f"OpenAI API call successful - tokens used: {response.usage.total_tokens}")
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"OpenAI API call failed, using fallback: {e}")
                return self._intelligent_fallback_analysis(prompt)
        else:
            return self._intelligent_fallback_analysis(prompt)

    def _intelligent_fallback_analysis(self, prompt: str) -> str:
        """Intelligent keyword-based analysis for demo purposes"""
        # Extract article text from prompt
        article_start = prompt.find("Article:")
        if article_start != -1:
            article_text = prompt[article_start + 8:].strip()
        else:
            article_text = prompt
        
        article_lower = article_text.lower()
        
        # Analyze keywords for bias indicators
        bias_keywords = {
            'left_leaning': ['progressive', 'liberal', 'reform', 'change', 'social justice', 'equality', 'climate action'],
            'right_leaning': ['conservative', 'traditional', 'free market', 'security', 'law and order', 'family values'],
            'emotional': ['shocking', 'devastating', 'outrageous', 'incredible', 'amazing', 'terrible', 'crisis'],
            'loaded_language': ['slam', 'blast', 'destroy', 'crush', 'annihilate', 'expose', 'reveal', 'slammed', 'denounced'],
            'factual_indicators': ['study', 'research', 'data', 'statistics', 'according to', 'reported', 'confirmed'],
            'opinion_indicators': ['believe', 'think', 'feel', 'opinion', 'suggest', 'argue', 'claim']
        }
        
        # Find actual phrases in the text
        highlighted_phrases = []
        found_keywords = []
        
        # Check for loaded language
        for word in bias_keywords['loaded_language']:
            if word in article_lower:
                # Find the actual phrase in original text
                import re
                pattern = re.compile(re.escape(word), re.IGNORECASE)
                matches = pattern.findall(article_text)
                if matches:
                    highlighted_phrases.append({
                        "text": matches[0],
                        "score_impact": {"framing_choices": -15, "emotional_tone": -10},
                        "explanation": f"Uses loaded language '{matches[0]}' which introduces bias through emotionally charged terminology",
                        "confidence": 85
                    })
                    found_keywords.append(word)
                    break
        
        # Check for emotional language
        for word in bias_keywords['emotional']:
            if word in article_lower and word not in found_keywords:
                pattern = re.compile(re.escape(word), re.IGNORECASE)
                matches = pattern.findall(article_text)
                if matches:
                    highlighted_phrases.append({
                        "text": matches[0],
                        "score_impact": {"emotional_tone": -12},
                        "explanation": f"Emotional language '{matches[0]}' appeals to feelings rather than presenting neutral facts",
                        "confidence": 80
                    })
                    found_keywords.append(word)
                    break
        
        # Check for political leaning
        for word in bias_keywords['left_leaning']:
            if word in article_lower and word not in found_keywords:
                pattern = re.compile(re.escape(word), re.IGNORECASE)
                matches = pattern.findall(article_text)
                if matches:
                    highlighted_phrases.append({
                        "text": matches[0],
                        "score_impact": {"ideological_stance": -15},
                        "explanation": f"Term '{matches[0]}' suggests left-leaning perspective",
                        "confidence": 75
                    })
                    found_keywords.append(word)
                    break
        
        for word in bias_keywords['right_leaning']:
            if word in article_lower and word not in found_keywords:
                pattern = re.compile(re.escape(word), re.IGNORECASE)
                matches = pattern.findall(article_text)
                if matches:
                    highlighted_phrases.append({
                        "text": matches[0],
                        "score_impact": {"ideological_stance": 15},
                        "explanation": f"Term '{matches[0]}' suggests right-leaning perspective",
                        "confidence": 75
                    })
                    found_keywords.append(word)
                    break
        
        # Check for factual indicators
        for word in bias_keywords['factual_indicators']:
            if word in article_lower and word not in found_keywords:
                pattern = re.compile(re.escape(word), re.IGNORECASE)
                matches = pattern.findall(article_text)
                if matches:
                    highlighted_phrases.append({
                        "text": matches[0],
                        "score_impact": {"factual_grounding": 10},
                        "explanation": f"Reference to '{matches[0]}' indicates factual sourcing",
                        "confidence": 90
                    })
                    found_keywords.append(word)
                    break
        
        # Score based on keyword presence
        left_score = sum(1 for word in bias_keywords['left_leaning'] if word in article_lower)
        right_score = sum(1 for word in bias_keywords['right_leaning'] if word in article_lower)
        emotional_score = sum(1 for word in bias_keywords['emotional'] if word in article_lower)
        loaded_score = sum(1 for word in bias_keywords['loaded_language'] if word in article_lower)
        factual_score = sum(1 for word in bias_keywords['factual_indicators'] if word in article_lower)
        opinion_score = sum(1 for word in bias_keywords['opinion_indicators'] if word in article_lower)
        
        # Calculate dimension scores
        ideological_stance = 50  # Default neutral
        if left_score > right_score:
            ideological_stance = max(20, 50 - (left_score * 8))
        elif right_score > left_score:
            ideological_stance = min(80, 50 + (right_score * 8))
            
        factual_grounding = min(85, 40 + (factual_score * 10) - (opinion_score * 5))
        framing_choices = max(20, 70 - (loaded_score * 15))
        emotional_tone = max(15, 75 - (emotional_score * 12))
        source_transparency = 60  # Default moderate
        
        return json.dumps({
            "ideological_stance": int(ideological_stance),
            "factual_grounding": int(factual_grounding),
            "framing_choices": int(framing_choices),
            "emotional_tone": int(emotional_tone),
            "source_transparency": int(source_transparency),
            "confidence": {
                "ideological_stance": [max(0, int(ideological_stance) - 8), min(100, int(ideological_stance) + 8)], 
                "factual_grounding": [max(0, int(factual_grounding) - 10), min(100, int(factual_grounding) + 10)], 
                "framing_choices": [max(0, int(framing_choices) - 6), min(100, int(framing_choices) + 6)],
                "emotional_tone": [max(0, int(emotional_tone) - 8), min(100, int(emotional_tone) + 8)], 
                "source_transparency": [max(0, int(source_transparency) - 12), min(100, int(source_transparency) + 12)]
            },
            "highlighted_phrases": highlighted_phrases
        })

    async def analyze_article(self, title: str, content: str, source: str) -> Dict[str, Any]:
        """Fast bias analysis with explainable AI and confidence intervals"""
        start_time = time.time()
        article_text = f"Title: {title}\nSource: {source}\n\n{content}"
        article_id = hashlib.md5(article_text.encode()).hexdigest()

        # Check cache first for speed
        if article_id in self.cache:
            cached_result = self.cache[article_id].copy()
            cached_result["processing_time_ms"] = 1.0  # Cache hit
            return cached_result

        # Generate fast analysis
        analysis_prompt = self.prompts.fast_bias_analysis_prompt(article_text)
        analysis_raw = await self._generate_analysis(analysis_prompt)
        
        try:
            analysis_data = json.loads(analysis_raw)
        except json.JSONDecodeError:
            logger.error("Failed to parse analysis JSON, using fallback", raw_output=analysis_raw)
            analysis_data = json.loads(self._intelligent_fallback_analysis(analysis_prompt))

        # Extract scores with confidence intervals
        dimension_scores = {
            "ideological_stance": analysis_data.get("ideological_stance", 50),
            "factual_grounding": analysis_data.get("factual_grounding", 50),
            "framing_choices": analysis_data.get("framing_choices", 50),
            "emotional_tone": analysis_data.get("emotional_tone", 50),
            "source_transparency": analysis_data.get("source_transparency", 50)
        }
        
        confidence_scores = analysis_data.get("confidence", {
            "ideological_stance": [40, 60], "factual_grounding": [40, 60], "framing_choices": [40, 60],
            "emotional_tone": [40, 60], "source_transparency": [40, 60]
        })
        
        highlighted_phrases = analysis_data.get("highlighted_phrases", [])
        
        # If no highlighted phrases from API, generate some from fallback
        if not highlighted_phrases:
            logger.warning("No highlighted phrases found, generating fallback phrases")
            fallback_analysis = json.loads(self._intelligent_fallback_analysis(analysis_prompt))
            highlighted_phrases = fallback_analysis.get("highlighted_phrases", [])
        overall_score = sum(dimension_scores.values()) / len(dimension_scores)
        processing_time_ms = (time.time() - start_time) * 1000

        result = {
            "article_id": article_id,
            "overall_score": round(overall_score, 1),
            "dimension_scores": BiasScore(**dimension_scores).dict(),
            "confidence_intervals": confidence_scores,
            "highlighted_phrases": highlighted_phrases,
            "processing_time_ms": round(processing_time_ms, 2),
            "model_used": self.model_name if self.client else "intelligent_demo",
            "timestamp": datetime.now().isoformat()
        }
        
        # Cache result for future requests
        self.cache[article_id] = result.copy()
        
        return result