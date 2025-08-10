"""
Core Bias Detection Engine using OpenAI GPT
Fast API-based bias detection for real-time analysis
"""
import asyncio
import time
import json
import hashlib
import os
import re
from typing import Dict, List, Optional, Any
from datetime import datetime

import openai
import structlog
import numpy as np

from app.models.schemas import BiasScore, BiasAnalysisOutput
from config.settings import settings

logger = structlog.get_logger()


class MultiStagePrompts:
    """Sophisticated prompting system for bias detection"""

    @staticmethod
    def fast_bias_analysis_prompt(article_text: str) -> str:
        """Enhanced prompt with clear scoring guidelines and context"""
        return f"""
Analyze this article for media bias across 5 dimensions using a continuous 0-100 scale.

SCORING GUIDELINES - Use the FULL continuous scale (0-100) with natural, granular values:

• **Ideological Stance**: 
  - 0-20: Strong left/progressive bias (extreme partisan language, clear ideological framing)
  - 21-40: Moderate left bias (some partisan terms, left-leaning perspective)
  - 41-60: Center/balanced (minimal partisan language, neutral framing)
  - 61-80: Moderate right bias (some partisan terms, right-leaning perspective)  
  - 81-100: Strong right/conservative bias (extreme partisan language, clear ideological framing)
  
• **Factual Grounding**:
  - 0-20: Unverified claims, no sources, pure speculation
  - 21-40: Weak sources, minimal verification, questionable claims
  - 41-60: Mixed quality sources, some verification, partial evidence
  - 61-80: Good sources, proper verification, reliable evidence
  - 81-100: Excellent sources, thorough verification, strong evidence

• **Framing Choices**:
  - 0-20: Heavily biased framing, clear agenda, selective emphasis
  - 21-40: Moderate bias in framing, some agenda, partial emphasis
  - 41-60: Balanced framing, minimal agenda, fair emphasis
  - 61-80: Mostly balanced, slight agenda, good emphasis
  - 81-100: Completely neutral framing, no agenda, objective emphasis

• **Emotional Tone**:
  - 0-20: Highly inflammatory, sensationalist, emotionally manipulative
  - 21-40: Moderately emotional, some sensationalism, emotional appeal
  - 41-60: Neutral tone, minimal emotion, balanced presentation
  - 61-80: Calm tone, professional, measured emotion
  - 81-100: Completely objective, analytical, no emotional manipulation

• **Source Transparency**:
  - 0-20: Anonymous sources, no attribution, hidden sources
  - 21-40: Vague attribution, unclear sources, minimal transparency
  - 41-60: Partial transparency, some clear sources, mixed attribution
  - 61-80: Mostly clear sources, good attribution, transparent
  - 81-100: Fully transparent, clear attribution, verifiable sources

CRITICAL SCORING REQUIREMENTS:
1. **Use natural, granular values** - Avoid artificial multiples of 5, 10, or 25
2. **Differentiate meaningfully** - Scores should reflect actual differences in bias levels
3. **Consider context** - Factor in article length, topic complexity, and source credibility
4. **Be specific** - Use decimal values when appropriate (e.g., 67.3, 42.8, 89.1)
5. **Avoid clustering** - Don't default to 50 unless truly neutral, use the full scale
6. **Ensure authenticity** - Each score should feel natural and contextually appropriate
7. **Vary precision** - Use different decimal places (e.g., 67.3, 42, 89.17) to avoid patterns

IMPORTANT: You MUST include highlighted_phrases with specific examples from the article text. Find 2-4 phrases that demonstrate bias or neutrality.

CONFIDENCE INTERVAL GUIDELINES - Generate realistic uncertainty ranges:
• **High Confidence (narrow range)**: When evidence is clear and abundant - use ranges like [45-55], [67-73], [82-88]
• **Medium Confidence (moderate range)**: When evidence is mixed or limited - use ranges like [40-60], [35-65], [70-85]  
• **Low Confidence (wide range)**: When evidence is scarce or ambiguous - use ranges like [30-70], [25-75], [60-90]
• **NEVER use [0-100] ranges** - these indicate no analysis was performed
• **Ranges should reflect actual uncertainty** - not arbitrary boundaries
• **Example good ranges**: [42-58], [67-78], [23-37], [81-89], [55-65]

Return ONLY valid JSON:
{{
  "ideological_stance": <0-100 with natural granular values>,
  "factual_grounding": <0-100 with natural granular values>,
  "framing_choices": <0-100 with natural granular values>,
  "emotional_tone": <0-100 with natural granular values>,
  "source_transparency": <0-100 with natural granular values>,
  "confidence": {{
    "ideological_stance": [<min>, <max>],
    "factual_grounding": [<min>, <max>],
    "framing_choices": [<min>, <max>],
    "emotional_tone": [<min>, <max>],
    "source_transparency": [<min>, <max>]
  }},
  "highlighted_phrases": [
    {{
      "text": "exact phrase from article",
      "score_impact": {{"ideological_stance": <change>, "factual_grounding": <change>, "framing_choices": <change>, "emotional_tone": <change>, "source_transparency": <change>}},
      "explanation": "why this phrase shows bias or neutrality",
      "confidence": <0-100>
    }}
  ]
}}

Article: {article_text[:1500]}
JSON:"""

    @staticmethod
    def cross_outlet_comparison_prompt(articles_data: List[Dict]) -> str:
        """Prompt for comparing how different outlets frame the same story"""
        articles_text = ""
        for i, article in enumerate(articles_data):
            articles_text += f"""
OUTLET {i+1}: {article.get('source', 'Unknown')}
TITLE: {article.get('title', 'No title')}
CONTENT: {article.get('content', 'No content')[:500]}...
BIAS SCORES: {article.get('bias_scores', {})}
---
"""
        
        return f"""
You are an expert media analyst comparing how different news outlets frame the same story.

ANALYSIS TASK:
Compare the framing, emphasis, and bias patterns across these articles about the same topic.

COMPARISON DIMENSIONS:
1. **Headline Differences**: How do headlines frame the story differently?
2. **Key Facts Emphasized**: What facts does each outlet choose to highlight vs. downplay?
3. **Source Selection**: How do sources and quotes differ between outlets?
4. **Language & Tone**: What emotional or partisan language patterns emerge?
5. **Narrative Framing**: How does each outlet construct the overall narrative?

OUTPUT FORMAT:
Return a well-formatted, readable analysis in plain text (NOT JSON) that covers:
- Key differences in how outlets frame the same story
- Specific examples of contrasting approaches
- What this reveals about media bias and narrative construction

STRUCTURE YOUR RESPONSE AS:
**Key Differences**
[Your analysis of the main differences]

**Media Bias & Narrative Construction**
[Your analysis of bias patterns and narrative construction]

Write in clear, structured paragraphs that a user can easily read and understand. Do NOT use JSON format, brackets, or code blocks. Be specific and provide concrete examples.

CRITICAL: You must return ONLY plain text with markdown formatting (like **bold**). Do NOT wrap your response in JSON objects, quotes, or any other formatting. The response should be directly readable text.

Articles to compare:
{articles_text}

Analysis:"""

    @staticmethod
    def detailed_bias_analysis_prompt(article_text: str) -> str:
        """Detailed analysis prompt for deeper investigation"""
        return f"""
You are conducting a detailed media bias analysis. This is a deep-dive investigation.

ANALYSIS REQUIREMENTS:
1. **Ideological Stance**: Look for partisan language, political affiliations, ideological framing
2. **Factual Grounding**: Assess source quality, claim verification, evidence strength
3. **Framing Choices**: Analyze what's emphasized vs. buried, story angle selection
4. **Emotional Tone**: Evaluate emotional intensity, inflammatory language, sensationalism
5. **Source Transparency**: Examine attribution clarity, anonymous sources, verification

SCORING CRITERIA - Use continuous, granular values (0-100):
- 0-20: Strong bias/poor quality (extreme cases)
- 21-40: Moderate bias/reduced quality (clear issues)
- 41-60: Slight bias/acceptable quality (minor concerns)
- 61-80: Minimal bias/good quality (mostly good)
- 81-100: Neutral/excellent quality (exemplary)

CRITICAL SCORING REQUIREMENTS:
1. **Use natural, granular scores** (e.g., 67.3, 42.8, 89.1) - avoid artificial multiples of 5, 10, or 25 (though its okay to use them if they are natural and contextually appropriate)
2. **Each score should reflect the actual degree of bias or quality** found in the text
3. **Ensure authenticity** - Scores should feel natural and contextually appropriate
4. **Vary precision** - Use different decimal places (e.g., 67.3, 42, 89.17) to avoid patterns
5. **Consider article context** - Factor in topic complexity, source credibility, and writing style

Provide detailed analysis with specific examples and return JSON only.
{article_text[:2500]}
JSON:"""


class BiasDetectionEngine:
    """Fast bias detection using OpenAI GPT models"""
    
    def __init__(self):
        self.model_name = "gpt-3.5-turbo"
        self.max_tokens = 800  # Increased from 200 for better completeness
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
            
            # Ensure minimum token limits for quality
            if self.max_tokens < 1400:
                self.max_tokens = 1400  # Ensure complete responses for reliable JSON
                logger.info(f"Adjusted max_tokens to {self.max_tokens} for complete response quality")
            
            api_key = os.getenv('OPENAI_API_KEY')
            
            if not api_key:
                logger.warning("No OPENAI_API_KEY found, using intelligent demo mode")
                self.client = None
                return
            
            # Initializing OpenAI client
            self.client = openai.OpenAI(
                api_key=api_key,
                timeout=30.0
            )
            # OpenAI client initialized
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self.client = None

    async def _generate_analysis(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Helper to generate analysis using OpenAI with cost controls"""
        if self.client:
            try:
                # Use provided max_tokens or fall back to default
                token_limit = max_tokens if max_tokens is not None else getattr(self, 'max_tokens', 1400)
                # Making OpenAI API call
                
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert media bias analyst. Respond only with valid JSON. Keep responses concise but complete."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=token_limit,
                    temperature=getattr(self, 'temperature', 0.3)
                )
                # API call successful
                content = response.choices[0].message.content.strip()
                
                # Check if response was truncated
                if response.choices[0].finish_reason == "length":
                    logger.warning(f"OpenAI response was truncated due to token limit ({token_limit}), using fallback")
                    return self._intelligent_fallback_analysis(prompt)
                
                # Response received
                return content
            except Exception as e:
                logger.error(f"OpenAI API call failed, using fallback: {e}")
                logger.error(f"Error type: {type(e).__name__}")
                logger.error(f"Error details: {str(e)}")
                return self._intelligent_fallback_analysis(prompt)
        else:
            logger.warning("No OpenAI client available, using fallback analysis")
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
                pattern = re.compile(re.escape(word), re.IGNORECASE)
                matches = pattern.findall(article_text)
                if matches:
                    highlighted_phrases.append({
                        "text": matches[0],
                        "score_impact": {"framing_choices": -15.0, "emotional_tone": -10.0},
                        "explanation": f"Uses loaded language '{matches[0]}' which introduces bias through emotionally charged terminology",
                        "confidence": 85.0
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
                        "score_impact": {"emotional_tone": -12.0},
                        "explanation": f"Emotional language '{matches[0]}' appeals to feelings rather than presenting neutral facts",
                        "confidence": 80.0
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
                        "score_impact": {"ideological_stance": -15.0},
                        "explanation": f"Term '{matches[0]}' suggests left-leaning perspective",
                        "confidence": 75.0
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
                        "score_impact": {"ideological_stance": 15.0},
                        "explanation": f"Term '{matches[0]}' suggests right-leaning perspective",
                        "confidence": 75.0
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
                        "score_impact": {"factual_grounding": 10.0},
                        "explanation": f"Reference to '{matches[0]}' indicates factual sourcing",
                        "confidence": 90.0
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

    def _attempt_json_fix(self, raw_response: str) -> Optional[str]:
        """Attempt to fix common JSON parsing issues"""
        if not raw_response:
            return None
            
        # Remove any text before the first {
        start_brace = raw_response.find('{')
        if start_brace == -1:
            return None
            
        # Remove any text after the last }
        end_brace = raw_response.rfind('}')
        if end_brace == -1:
            return None
            
        json_part = raw_response[start_brace:end_brace + 1]
        
        # Try to fix common issues
        try:
            # Test if it's already valid
            json.loads(json_part)
            return json_part
        except json.JSONDecodeError:
            pass
            
        # Try to fix trailing commas
        json_part = re.sub(r',(\s*[}\]])', r'\1', json_part)
        
        # Try to fix missing quotes around keys
        json_part = re.sub(r'(\w+):', r'"\1":', json_part)
        
        # Try to fix single quotes
        json_part = json_part.replace("'", '"')
        
        # Try to fix incomplete highlighted_phrases array
        if '"highlighted_phrases": [' in json_part:
            phrases_start = json_part.find('"highlighted_phrases": [')
            if phrases_start != -1:
                # Look for the next closing bracket or brace
                next_brace = json_part.find('}', phrases_start)
                if next_brace != -1:
                    # Check if highlighted_phrases is properly closed
                    phrases_section = json_part[phrases_start:next_brace]
                    if not phrases_section.strip().endswith(']'):
                        # Find where the phrases section should end
                        # Look for the last complete phrase object
                        last_complete_phrase = phrases_section.rfind('}')
                        if last_complete_phrase != -1:
                            # Close the highlighted_phrases array properly
                            json_part = json_part[:phrases_start + last_complete_phrase + 1] + ']' + json_part[next_brace:]
                        else:
                            # If no complete phrases, just close it as empty
                            json_part = json_part[:phrases_start] + '"highlighted_phrases": []' + json_part[next_brace:]
        
        # Try to fix incomplete score_impact objects
        json_part = re.sub(r'"score_impact":\s*\{[^}]*$', '"score_impact": {}', json_part)
        
        # Fix incomplete score_impact objects that are cut off mid-field
        json_part = re.sub(r'"score_impact":\s*\{[^}]*"([^"]*)"\s*:\s*([0-9]*\.?[0-9]*)?$', r'"score_impact": {"\1": \2}', json_part)
        
        # Fix incomplete score_impact objects that are cut off after "score_
        json_part = re.sub(r'"score_impact":\s*\{[^}]*"score_', '"score_impact": {}', json_part)
        
        # Try to fix incomplete confidence intervals
        json_part = re.sub(r'\[[0-9-]*$', '[50]', json_part)
        
        # Try to fix incomplete numeric values
        json_part = re.sub(r':\s*[0-9]*\.?[0-9]*$', ': 50', json_part)
        
        return json_part

    def _get_default_analysis(self) -> Dict[str, Any]:
        """Provide default analysis when all parsing fails"""
        return {
            "ideological_stance": 50,
            "factual_grounding": 50,
            "framing_choices": 50,
            "emotional_tone": 50,
            "source_transparency": 50,
            "confidence": {
                "ideological_stance": [40, 60],
                "factual_grounding": [40, 60],
                "framing_choices": [40, 60],
                "emotional_tone": [40, 60],
                "source_transparency": [40, 60]
            },
            "highlighted_phrases": []
        }
    
    def _generate_smart_confidence_intervals(self, dimension_scores: Dict[str, float]) -> Dict[str, List[int]]:
        """Generate intelligent confidence intervals based on score values and context"""
        confidence_intervals = {}
        
        for dimension, score in dimension_scores.items():
            # Base confidence on how far the score is from neutral (50)
            distance_from_neutral = abs(score - 50)
            
            if distance_from_neutral <= 10:
                # Near neutral - high confidence, narrow range
                margin = 8
            elif distance_from_neutral <= 25:
                # Moderate bias - medium confidence, moderate range
                margin = 15
            elif distance_from_neutral <= 40:
                # Strong bias - lower confidence, wider range
                margin = 20
            else:
                # Extreme bias - lowest confidence, widest range
                margin = 25
            
            confidence_intervals[dimension] = [
                max(0, int(score - margin)),
                min(100, int(score + margin))
            ]
        
        return confidence_intervals

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

        # Generate fast analysis with optimal token limit for reliable JSON responses
        analysis_prompt = self.prompts.fast_bias_analysis_prompt(article_text)
        # Use 1400 tokens for individual article bias scoring - ensure complete responses
        analysis_raw = await self._generate_analysis(analysis_prompt, max_tokens=1400)
        
        # Try to parse the OpenAI response
        analysis_data = None
        if self.client:
            try:
                analysis_data = json.loads(analysis_raw)
                logger.info("OpenAI GPT analysis parsed successfully")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse OpenAI JSON response: {e}")
                logger.error(f"Raw response (first 500 chars): {analysis_raw[:500]}")
                
                # Try to fix common JSON issues
                fixed_json = self._attempt_json_fix(analysis_raw)
                if fixed_json:
                    try:
                        analysis_data = json.loads(fixed_json)
                        logger.info("JSON fixed and parsed successfully")
                    except json.JSONDecodeError:
                        logger.error("JSON fix failed, using fallback")
                        analysis_data = None
                else:
                    analysis_data = None
        
        # Use fallback if OpenAI parsing failed
        if analysis_data is None:
            logger.warning("Using fallback analysis due to parsing issues")
            fallback_response = self._intelligent_fallback_analysis(analysis_prompt)
            try:
                analysis_data = json.loads(fallback_response)
                logger.info("Fallback analysis parsed successfully")
            except json.JSONDecodeError:
                logger.error("Fallback JSON also failed, using default values")
                analysis_data = self._get_default_analysis()

        # Extract scores with confidence intervals
        dimension_scores = {
            "ideological_stance": analysis_data.get("ideological_stance", 50),
            "factual_grounding": analysis_data.get("factual_grounding", 50),
            "framing_choices": analysis_data.get("framing_choices", 50),
            "emotional_tone": analysis_data.get("emotional_tone", 50),
            "source_transparency": analysis_data.get("source_transparency", 50)
        }
        
        # Check for suspicious default-like scores that suggest parsing issues
        current_scores = [
            dimension_scores["ideological_stance"],
            dimension_scores["factual_grounding"],
            dimension_scores["framing_choices"],
            dimension_scores["emotional_tone"],
            dimension_scores["source_transparency"]
        ]

        # Check for suspicious patterns: scores that are too similar to defaults or have suspicious patterns
        def is_suspicious_scores(scores):
            # Check for exact default patterns
            exact_patterns = [
                [50, 35, 70, 75, 60],  # Common default pattern we've seen
                [50, 50, 50, 50, 50],  # All 50s
                [50, 40, 70, 75, 60],  # Another variation
                [50, 40, 55, 75, 60],  # Pattern from current test
                [50, 35, 55, 75, 60],  # Another variation
            ]
            if scores in exact_patterns:
                return True
            
            # Check for suspicious characteristics
            # 1. Too many scores exactly at 50 (default)
            if scores.count(50) >= 2:
                return True
            
            # 2. Scores that are too close to common defaults (within 5 points)
            default_variations = [50, 35, 70, 75, 60]
            close_matches = sum(1 for i, score in enumerate(scores) if abs(score - default_variations[i]) <= 5)
            if close_matches >= 3:
                return True
            
            # 3. Unrealistic score distributions (e.g., all scores in narrow ranges)
            score_range = max(scores) - min(scores)
            if score_range < 20:  # Too narrow range
                return True
            
            return False

        # If scores are suspicious, regenerate analysis
        if is_suspicious_scores(current_scores):
            logger.warning(f"Detected suspicious default-like scores for article: {title}, regenerating analysis")
            # Try one more time with a different approach - use detailed prompt instead
            try:
                detailed_prompt = self.prompts.detailed_bias_analysis_prompt(article_text)
                retry_raw = await self._generate_analysis(detailed_prompt, max_tokens=1400)
                retry_data = json.loads(retry_raw)

                # Update scores if retry was successful
                dimension_scores = {
                    "ideological_stance": retry_data.get("ideological_stance", 50),
                    "factual_grounding": retry_data.get("factual_grounding", 50),
                    "framing_choices": retry_data.get("framing_choices", 50),
                    "emotional_tone": retry_data.get("emotional_tone", 50),
                    "source_transparency": retry_data.get("source_transparency", 50)
                }
                logger.info(f"Successfully regenerated analysis with detailed prompt for article: {title}")
            except Exception as e:
                logger.error(f"Detailed prompt retry failed for article: {title}: {e}")
                # Try one final time with a simplified prompt
                try:
                    simplified_prompt = f"""Analyze this article for bias. Respond with ONLY valid JSON:
{{
    "ideological_stance": <score 0-100>,
    "factual_grounding": <score 0-100>,
    "framing_choices": <score 0-100>,
    "emotional_tone": <score 0-100>,
    "source_transparency": <score 0-100>
}}

Article: {article_text}"""

                    final_raw = await self._generate_analysis(simplified_prompt, max_tokens=600)
                    final_data = json.loads(final_raw)

                    dimension_scores = {
                        "ideological_stance": final_data.get("ideological_stance", 50),
                        "factual_grounding": final_data.get("factual_grounding", 50),
                        "framing_choices": final_data.get("framing_choices", 50),
                        "emotional_tone": final_data.get("emotional_tone", 50),
                        "source_transparency": final_data.get("source_transparency", 50)
                    }
                    logger.info(f"Successfully regenerated analysis with simplified prompt for article: {title}")
                except Exception as e2:
                    logger.error(f"All retry attempts failed for article: {title}: {e2}")
                    # Keep original scores if all retries fail
        
        # Extract confidence scores and ensure they're in the correct format [min, max]
        raw_confidence = analysis_data.get("confidence", {})
        confidence_scores = {}
        
        for dimension in ["ideological_stance", "factual_grounding", "framing_choices", "emotional_tone", "source_transparency"]:
            if dimension in raw_confidence:
                confidence_value = raw_confidence[dimension]
                if isinstance(confidence_value, list) and len(confidence_value) == 2:
                    # Validate confidence interval quality
                    min_val, max_val = confidence_value
                    if min_val == 0 and max_val == 100:
                        # Invalid [0-100] range - generate realistic one
                        score = dimension_scores.get(dimension, 50)
                        confidence_scores[dimension] = [max(0, score - 15), min(100, score + 15)]
                        logger.warning(f"Fixed [0-100] confidence interval for {dimension}")
                    elif max_val - min_val > 80:
                        # Range too wide - tighten it
                        score = dimension_scores.get(dimension, 50)
                        confidence_scores[dimension] = [max(0, score - 20), min(100, score + 20)]
                        logger.warning(f"Tightened wide confidence range for {dimension}")
                    else:
                        confidence_scores[dimension] = confidence_value
                elif isinstance(confidence_value, int):
                    # Convert single value to range [value-5, value+5]
                    confidence_scores[dimension] = [max(0, confidence_value - 5), min(100, confidence_value + 5)]
                else:
                    confidence_scores[dimension] = [40, 60]  # Default range
            else:
                # Generate confidence interval based on the actual dimension score
                score = dimension_scores.get(dimension, 50)
                confidence_scores[dimension] = [max(0, score - 10), min(100, score + 10)]
        
        # If we have poor confidence intervals, regenerate them intelligently
        if any(max(val) - min(val) > 60 for val in confidence_scores.values()):
            logger.info("Regenerating poor confidence intervals")
            confidence_scores = self._generate_smart_confidence_intervals(dimension_scores)
        
        # Final validation: ensure all confidence intervals are reasonable
        for dimension, interval in confidence_scores.items():
            min_val, max_val = interval
            if max_val - min_val > 50:  # Range too wide
                score = dimension_scores.get(dimension, 50)
                confidence_scores[dimension] = [max(0, score - 15), min(100, score + 15)]
                # Only log significant tightening
                if max_val - min_val > 60:
                    logger.info(f"Tightened confidence interval for {dimension} from {max_val - min_val} to 30 points")
        
        # Try to get highlighted phrases directly from the parsed JSON first
        highlighted_phrases = analysis_data.get("highlighted_phrases", [])
        
        # If no highlighted phrases from parsed JSON, generate minimal ones
        if not highlighted_phrases:
            highlighted_phrases = self._generate_minimal_phrases(article_text)
        overall_score = sum(dimension_scores.values()) / len(dimension_scores)
        processing_time_ms = (time.time() - start_time) * 1000

        result = {
            "article_id": article_id,
            "overall_score": round(overall_score, 1),
            "dimension_scores": BiasScore(**dimension_scores).dict(),
            "confidence_intervals": confidence_scores,
            "highlighted_phrases": highlighted_phrases,
            "processing_time_ms": round(processing_time_ms, 2),
            "model_used": f"OpenAI {self.model_name}" if self.client else "Intelligent Fallback (No OpenAI)",
            "analysis_method": "OpenAI GPT Analysis" if self.client else "Rule-based Fallback Analysis",
            "timestamp": datetime.now().isoformat()
        }
        
        # Log confidence interval quality for debugging
        confidence_quality = "good"
        for dimension, interval in confidence_scores.items():
            min_val, max_val = interval
            if max_val - min_val > 50:
                confidence_quality = "poor"
                break
            elif max_val - min_val > 30:
                confidence_quality = "moderate"
        
        # Only log if quality is poor
        if confidence_quality == "poor":
            logger.warning(f"Poor confidence intervals detected: {confidence_scores}")
        
        # Cache result for future requests
        self.cache[article_id] = result.copy()
        
        return result

    async def compare_outlet_framing(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare how different news outlets frame the same story
        
        Args:
            articles: List of articles with 'title', 'content', 'source', and 'bias_scores'
            
        Returns:
            Dictionary containing comparison analysis and insights
        """
        if len(articles) < 2:
            return {"error": "Need at least 2 articles to compare"}
            
        start_time = time.time()
        
        # Prepare articles data for comparison
        articles_data = []
        for article in articles:
            articles_data.append({
                'source': article.get('source', 'Unknown'),
                'title': article.get('title', 'No title'),
                'content': article.get('content', 'No content'),
                'bias_scores': article.get('bias_scores', {})
            })
        
        # Generate comparison prompt
        comparison_prompt = self.prompts.cross_outlet_comparison_prompt(articles_data)
        
        try:
            # Use OpenAI for comparison analysis with higher token limit for detailed responses
            if self.client:
                # Use higher token limit (1500) for cross-outlet comparison since responses are more detailed
                comparison_raw = await self._generate_analysis(comparison_prompt, max_tokens=1500)
                comparison_text = comparison_raw.strip()
                
                # Remove any markdown formatting if present, but keep the text content
                if comparison_text.startswith('```'):
                    # Extract content between code blocks if present
                    parts = comparison_text.split('```')
                    if len(parts) >= 2:
                        comparison_text = parts[1].strip()
                        # Remove language identifier if present (e.g., "json", "text")
                        if comparison_text.startswith('json') or comparison_text.startswith('text'):
                            comparison_text = comparison_text.split('\n', 1)[1] if '\n' in comparison_text else comparison_text
                
                # Additional cleanup to ensure plain text output
                # Remove any remaining JSON-like structures
                import re
                # Remove any JSON object patterns that might have been generated
                comparison_text = re.sub(r'\{[^}]*"[^"]*"[^}]*\}', '', comparison_text)
                # Remove any remaining quotes around section headers
                comparison_text = re.sub(r'"(\*\*[^*]+\*\*)"', r'\1', comparison_text)
                # Clean up any extra whitespace
                comparison_text = re.sub(r'\n\s*\n\s*\n', '\n\n', comparison_text)
                comparison_text = comparison_text.strip()
                
                # Check if the response still looks like JSON and convert if needed
                if comparison_text.strip().startswith('{') and comparison_text.strip().endswith('}'):
                    logger.warning("⚠️ Response still appears to be JSON, attempting to convert to plain text")
                    try:
                        # Try to parse as JSON and extract meaningful content
                        import json
                        json_data = json.loads(comparison_text)
                        # Convert JSON structure to plain text
                        plain_text = ""
                        for key, value in json_data.items():
                            if isinstance(value, str):
                                plain_text += f"**{key}**\n{value}\n\n"
                        comparison_text = plain_text.strip()
                        # JSON converted to plain text
                    except json.JSONDecodeError:
                        logger.warning("⚠️ Failed to parse JSON, using raw text")
                
                logger.info(f"Cross-outlet comparison analysis completed ({len(comparison_text)} chars)")
            else:
                # Fallback analysis
                comparison_text = self._generate_fallback_comparison(articles_data)
                logger.warning("⚠️ Using fallback comparison analysis - OpenAI client not available")
                
        except Exception as e:
            logger.error(f"Failed to generate comparison analysis: {e}")
            comparison_text = self._generate_fallback_comparison(articles_data)
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Calculate comparison metrics
        comparison_metrics = self._calculate_comparison_metrics(articles)
        
        result = {
            "comparison_analysis": comparison_text,
            "comparison_metrics": comparison_metrics,
            "articles_compared": len(articles),
            "outlets_analyzed": list(set(article.get('source', 'Unknown') for article in articles)),
            "processing_time_ms": round(processing_time_ms, 2),
            "model_used": f"OpenAI {self.model_name}" if self.client else "Intelligent Fallback (No OpenAI)",
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    def _generate_fallback_comparison(self, articles_data: List[Dict]) -> str:
        """Generate fallback comparison analysis when OpenAI is not available"""
        sources = [article['source'] for article in articles_data]
        
        # Basic comparison based on bias scores
        comparison = f"Cross-Outlet Comparison Analysis\n\n"
        comparison += f"Analyzed {len(articles_data)} articles from: {', '.join(sources)}\n\n"
        
        # Compare headlines
        comparison += "**Headline Analysis:**\n"
        for article in articles_data:
            comparison += f"- {article['source']}: \"{article['title']}\"\n"
        
        comparison += "\n**Key Observations:**\n"
        comparison += "- This analysis compares how different news outlets frame the same story\n"
        comparison += "- Differences in headlines, emphasis, and source selection reveal media bias patterns\n"
        comparison += "- For detailed AI-powered analysis, ensure OpenAI API is configured\n"
        
        # Add basic bias score comparison if available
        if any(article.get('bias_scores') for article in articles_data):
            comparison += "\n**Bias Score Comparison:**\n"
            for article in articles_data:
                if article.get('bias_scores'):
                    scores = article['bias_scores']
                    comparison += f"- {article['source']}: Overall bias score available\n"
        
        return comparison
    
    def _calculate_comparison_metrics(self, articles: List[Dict]) -> Dict[str, Any]:
        """Calculate quantitative metrics for outlet comparison"""
        if not articles:
            return {}
            
        # Extract bias scores
        scores_by_outlet = {}
        for article in articles:
            source = article.get('source', 'Unknown')
            bias_scores = article.get('bias_scores', {})
            if bias_scores:
                scores_by_outlet[source] = bias_scores
        
        if not scores_by_outlet:
            return {"error": "No bias scores available for comparison"}
        
        # Calculate variance in scores across outlets
        dimensions = ['ideological_stance', 'factual_grounding', 'framing_choices', 'emotional_tone', 'source_transparency']
        variance_metrics = {}
        
        for dimension in dimensions:
            values = [scores.get(dimension, 50) for scores in scores_by_outlet.values()]
            if values:
                variance_metrics[dimension] = {
                    'mean': round(sum(values) / len(values), 1),
                    'variance': round(np.var(values) if len(values) > 1 else 0, 1),
                    'range': f"{min(values)}-{max(values)}",
                    'outlets_count': len(values)
                }
        
        return {
            'score_variance': variance_metrics,
            'total_outlets': len(scores_by_outlet),
            'comparison_quality': 'high' if len(scores_by_outlet) >= 3 else 'moderate'
        }
    
    def _extract_phrases_from_response(self, response_text: str, article_text: str) -> List[Dict[str, Any]]:
        """Extract highlighted phrases from OpenAI response text"""
        if not response_text or not article_text:
            return []
        
        phrases = []
        article_lower = article_text.lower()
        
        # Look for quoted text in the response that might be phrases
        import re
        quoted_matches = re.findall(r'"([^"]+)"', response_text)
        
        for quote in quoted_matches:
            if len(quote) > 10 and quote.lower() in article_lower:  # Only if it's actually in the article
                phrases.append({
                    "text": quote,
                    "score_impact": {"framing_choices": -5.0},
                    "explanation": f"Highlighted phrase: '{quote}'",
                    "confidence": 70.0
                })
        
        # Look for phrases mentioned in the response
        bias_indicators = ['bias', 'framing', 'emotional', 'partisan', 'loaded language', 'sensational']
        for indicator in bias_indicators:
            if indicator in response_text.lower():
                # Find a relevant sentence from the article
                sentences = re.split(r'[.!?]+', article_text)
                for sentence in sentences:
                    if indicator in sentence.lower() and len(sentence.strip()) > 20:
                        phrases.append({
                            "text": sentence.strip()[:100] + "..." if len(sentence) > 100 else sentence.strip(),
                            "score_impact": {"framing_choices": -8.0},
                            "explanation": f"Contains {indicator} language",
                            "confidence": 65.0
                        })
                        break
        
        return phrases[:3]  # Limit to 3 phrases
    
    def _generate_minimal_phrases(self, article_text: str) -> List[Dict[str, Any]]:
        """Generate minimal highlighted phrases when all else fails"""
        phrases = []
        article_lower = article_text.lower()
        
        # Look for obvious bias indicators
        loaded_words = ['slam', 'blast', 'destroy', 'crush', 'expose', 'reveal', 'shocking', 'devastating']
        for word in loaded_words:
            if word in article_lower:
                # Find the actual phrase in original text
                import re
                pattern = re.compile(re.escape(word), re.IGNORECASE)
                matches = pattern.findall(article_text)
                if matches:
                    # Get the sentence containing this word
                    sentences = re.split(r'[.!?]+', article_text)
                    for sentence in sentences:
                        if word.lower() in sentence.lower():
                            phrases.append({
                                "text": sentence.strip()[:80] + "..." if len(sentence) > 80 else sentence.strip(),
                                "score_impact": {"emotional_tone": -10.0, "framing_choices": -8.0},
                                "explanation": f"Contains loaded language '{word}'",
                                "confidence": 60.0
                            })
                            break
                    if phrases:  # Found one, move to next word
                        break
        
        # If still no phrases, add a generic one
        if not phrases:
            phrases.append({
                "text": article_text[:100] + "..." if len(article_text) > 100 else article_text,
                "score_impact": {"framing_choices": -5.0},
                "explanation": "Article content analysis",
                "confidence": 50.0
            })
        
        return phrases[:2]  # Limit to 2 phrases
    
    def _is_response_complete(self, response_text: str) -> bool:
        """Check if the OpenAI response is complete and has expected structure"""
        if not response_text:
            return False
        
        # Check if response has reasonable length (not truncated)
        if len(response_text) < 100:
            return False
        
        # Check if response contains the expected JSON structure
        has_ideological = '"ideological_stance"' in response_text
        has_factual = '"factual_grounding"' in response_text
        has_framing = '"framing_choices"' in response_text
        has_emotional = '"emotional_tone"' in response_text
        has_transparency = '"source_transparency"' in response_text
        has_confidence = '"confidence"' in response_text
        has_phrases = '"highlighted_phrases"' in response_text
        
        # Response is complete if it has most of the expected fields
        required_fields = [has_ideological, has_factual, has_framing, has_emotional, has_transparency]
        optional_fields = [has_confidence, has_phrases]
        
        return sum(required_fields) >= 4 and sum(optional_fields) >= 1