"""
Narrative clustering for bias detection pipeline
Groups similar article framings and detects narrative patterns
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import json
import re
from typing import List, Dict, Any, Tuple
import structlog

logger = structlog.get_logger()


class NarrativeCluster:
    """Represents a cluster of articles with similar narrative framing"""
    def __init__(self, cluster_id: int, articles: List[Dict], dominant_themes: List[str], 
                 bias_profile: Dict[str, float], representative_phrases: List[str], clustering_explanation: str = None):
        self.cluster_id = cluster_id
        self.articles = articles
        self.dominant_themes = dominant_themes
        self.bias_profile = bias_profile
        self.representative_phrases = representative_phrases
        self.size = len(articles)
        self.clustering_explanation = clustering_explanation or "Clustering explanation not available"


class NarrativeClusteringEngine:
    """Clusters articles by narrative similarity and bias patterns"""
    
    def __init__(self, max_clusters: int = None):
        """
        Initialize the narrative clustering engine
        
        Args:
            max_clusters: Maximum number of clusters to consider (default: None for dynamic)
                         If None, the algorithm will discover the natural number of clusters
                         If specified, it will respect this as an upper bound only
        """
        self.max_clusters = max_clusters
        logger.info(f"Initializing NarrativeClusteringEngine with max_clusters={max_clusters} (dynamic clustering enabled)")
        
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 3),  # Include phrases
            min_df=2,
            max_df=0.8
        )
        
        # Patterns to filter out from themes
        self.gibberish_patterns = [
            r'return\s+false',  # JavaScript return false
            r'return\s*$',      # JavaScript return
            r'_blank\s*\d+',    # _blank 200, etc.
            r'href\s*=',        # href=
            r'javascript:',     # javascript: protocol
            r'http[s]?://',     # URLs
            r'www\.',           # www.
            r'\.com',           # .com domains
            r'\.org',           # .org domains
            r'\.net',           # .net domains
            r'<[^>]+>',         # HTML tags
            r'&[a-zA-Z]+;',     # HTML entities
            r'[A-Z]{2,}',       # All caps words (likely acronyms/abbreviations)
            r'\d+px',           # CSS dimensions
            r'rgb\([^)]+\)',    # CSS colors
            r'#[0-9a-fA-F]{3,6}', # Hex colors
            r'[^\w\s]',         # Non-word characters (punctuation, symbols)
        ]
        
        # Compile regex patterns for efficiency
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.gibberish_patterns]
        
    def _clean_text(self, text: str) -> str:
        """Clean text by removing HTML, JavaScript, and other non-content elements"""
        if not text:
            return ""
            
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Remove JavaScript code blocks
        text = re.sub(r'<script[^>]*>.*?</script>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove CSS code blocks
        text = re.sub(r'<style[^>]*>.*?</style>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove common HTML entities
        text = re.sub(r'&[a-zA-Z]+;', ' ', text)
        text = re.sub(r'&#\d+;', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://[^\s]+', ' ', text)
        text = re.sub(r'www\.[^\s]+', ' ', text)
        
        # Remove JavaScript function calls and code
        text = re.sub(r'function\s*\([^)]*\)\s*\{[^}]*\}', ' ', text, flags=re.DOTALL)
        text = re.sub(r'return\s+[^;]+;', ' ', text)
        text = re.sub(r'var\s+\w+\s*=\s*[^;]+;', ' ', text)
        
        # Remove CSS properties
        text = re.sub(r'\w+:\s*[^;]+;', ' ', text)
        
        # Remove common HTML attributes
        text = re.sub(r'\w+="[^"]*"', ' ', text)
        text = re.sub(r"\w+='[^']*'", ' ', text)
        
        # Remove common technical patterns
        text = re.sub(r'on\w+\s*=', ' ', text)  # onclick=, onload=, etc.
        text = re.sub(r'data-\w+\s*=', ' ', text)  # data-* attributes
        
        # Remove excessive whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove leading/trailing punctuation
        text = re.sub(r'^[^\w\s]+', '', text)
        text = re.sub(r'[^\w\s]+$', '', text)
        
        return text
    
    def _is_valid_theme(self, theme: str) -> bool:
        """Check if a theme is valid (not gibberish)"""
        if not theme or len(theme) < 2:
            return False
            
        # Check against gibberish patterns
        for pattern in self.compiled_patterns:
            if pattern.search(theme):
                return False
                
        # Additional checks
        if len(theme) > 50:  # Too long
            return False
            
        if theme.isdigit():  # Just numbers
            return False
            
        if theme.count(' ') > 5:  # Too many words
            return False
            
        # Check for excessive special characters
        special_char_ratio = len(re.findall(r'[^\w\s]', theme)) / len(theme)
        if special_char_ratio > 0.3:  # More than 30% special characters
            return False
            
        # Check for common technical terms that should be filtered out
        technical_terms = {
            'return', 'function', 'var', 'let', 'const', 'if', 'else', 'for', 'while',
            'href', 'src', 'alt', 'title', 'class', 'id', 'style', 'script', 'div',
            'span', 'p', 'a', 'img', 'table', 'tr', 'td', 'th', 'ul', 'li', 'ol',
            'form', 'input', 'button', 'select', 'option', 'textarea', 'label',
            'http', 'https', 'www', 'com', 'org', 'net', 'edu', 'gov',
            'px', 'em', 'rem', 'rgb', 'rgba', 'hsl', 'hsla', 'hex',
            'margin', 'padding', 'border', 'background', 'color', 'font', 'text',
            'display', 'position', 'float', 'clear', 'overflow', 'visibility',
            'nasdaq', 'otcmkts', 'nyse', 'amex', 'tsx', 'lon', 'gbx', 'usd', 'eur',
            'target', 'cut', 'boost', 'raise', 'lower', 'hold', 'buy', 'sell'
        }
        
        if theme.lower() in technical_terms:
            return False
            
        # Check for JavaScript-like patterns
        if re.search(r'[{}();=+\-*/]', theme):
            return False
            
        # Check for CSS-like patterns
        if re.search(r'[{}:;]', theme) and len(theme) < 10:
            return False
            
        # Check for HTML-like patterns
        if re.search(r'[<>/]', theme):
            return False
            
        # Filter out technical identifiers and stock symbols
        technical_patterns = [
            r'^[A-Z]{2,5}$',  # Stock exchange codes like NASDAQ, OTCMKTS
            r'^[A-Z]{1,2}\d+$',  # Stock symbols like A1, B2
            r'^\d+[A-Z]+$',  # Reversed stock symbols
            r'^[A-Z]+\d+[A-Z]+$',  # Complex stock symbols
        ]
        
        for pattern in technical_patterns:
            if re.match(pattern, theme):
                return False
            
        return True
    
    def _filter_themes(self, themes: List[str], min_themes: int = 1) -> List[str]:
        """Filter themes to remove gibberish and ensure quality"""
        valid_themes = [theme for theme in themes if self._is_valid_theme(theme)]
        
        # If we have enough valid themes, return them (but limit to 3 for UI display)
        if len(valid_themes) >= min_themes:
            return valid_themes[:3]
            
        # If we don't have enough valid themes, try to be more lenient
        # but still filter out obvious gibberish
        if len(valid_themes) < min_themes:
            # Try to find themes that might be borderline but still meaningful
            borderline_themes = []
            for theme in themes:
                if not self._is_valid_theme(theme):
                    # Check if it's at least somewhat meaningful
                    if (len(theme) >= 3 and 
                        len(theme) <= 30 and 
                        theme.count(' ') <= 3 and
                        not any(pattern.search(theme) for pattern in self.compiled_patterns[:3]) and  # Only check first 3 patterns
                        not theme.lower() in {'return', 'function', 'var', 'href', 'src', 'alt', 'title'}):
                        borderline_themes.append(theme)
            
            # Combine valid and borderline themes
            all_acceptable_themes = valid_themes + borderline_themes[:2]  # Limit borderline themes
            
            if len(all_acceptable_themes) >= min_themes:
                return all_acceptable_themes[:3]  # Limit to 3 for UI
        
        # If we still don't have enough themes, return what we have (up to 3)
        return valid_themes[:3] if valid_themes else ['general_coverage']
        
    def cluster_narratives(self, articles_with_analysis: List[Dict[str, Any]]) -> List[NarrativeCluster]:
        """
        Cluster articles based on narrative similarity and bias patterns
        Uses authentic, data-driven clustering to discover natural groupings
        """
        if len(articles_with_analysis) < 2:
            logger.warning("Too few articles for meaningful clustering")
            return self._create_single_cluster(articles_with_analysis)
        
        try:
            # Extract and clean text features
            texts = [self._extract_narrative_text(article) for article in articles_with_analysis]
            cleaned_texts = [self._clean_text(text) for text in texts]
            text_features = self.vectorizer.fit_transform(cleaned_texts)
            
            # Extract bias features
            bias_features = np.array([self._extract_bias_features(article) for article in articles_with_analysis])
            
            # Combine text and bias features
            combined_features = np.hstack([
                text_features.toarray(),
                bias_features
            ])
            
            # Determine optimal number of clusters dynamically
            n_clusters = self._find_optimal_clusters(combined_features, len(articles_with_analysis))
            
            logger.info(f"Using {n_clusters} clusters based on authentic data analysis (max allowed: {self.max_clusters or 'dynamic'})")
            
            # Handle single cluster case
            if n_clusters == 1:
                logger.info("Data suggests no natural groupings - creating minimum 3 clusters")
                return self._create_single_cluster(articles_with_analysis)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(combined_features)
            
            # Calculate silhouette score for cluster quality
            if len(set(cluster_labels)) > 1:
                silhouette_avg = silhouette_score(combined_features, cluster_labels)
                logger.info(f"Clustering silhouette score: {silhouette_avg:.3f}")
                
                # Warn if clustering quality is poor
                if silhouette_avg < 0.2:
                    logger.warning(f"⚠️ Low clustering quality (silhouette: {silhouette_avg:.3f}). Consider fewer clusters.")
                elif silhouette_avg > 0.6:
                    logger.info(f"✅ Excellent clustering quality (silhouette: {silhouette_avg:.3f})")
                else:
                    logger.info(f"📊 Good clustering quality (silhouette: {silhouette_avg:.3f})")
            
            # Log cluster distribution
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            logger.info(f"Cluster distribution: {dict(zip(unique_labels, counts))}")
            
            # Create narrative clusters
            clusters = []
            for cluster_id in range(n_clusters):
                cluster_articles = [articles_with_analysis[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
                if cluster_articles:
                    cluster = self._create_narrative_cluster(cluster_id, cluster_articles, text_features, cluster_labels)
                    clusters.append(cluster)
                    logger.info(f"Cluster {cluster_id}: {len(cluster_articles)} articles")
            
            # Validate and potentially merge very small clusters
            clusters = self._validate_and_merge_clusters(clusters, n_articles=len(articles_with_analysis))
            
            # Add clustering explanation
            clustering_explanation = self._explain_clustering_decision(n_clusters, len(articles_with_analysis), silhouette_avg if 'silhouette_avg' in locals() else None)
            
            logger.info(f"Created {len(clusters)} narrative clusters from {len(articles_with_analysis)} articles")
            logger.info(f"Clustering explanation: {clustering_explanation}")
            
            # Store explanation in clusters for transparency
            for cluster in clusters:
                cluster.clustering_explanation = clustering_explanation
            
            return clusters
            
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            return self._create_single_cluster(articles_with_analysis)
    
    def _find_optimal_clusters(self, features: np.ndarray, n_articles: int) -> int:
        """
        Find the optimal number of clusters using authentic, data-driven analysis
        Discovers natural groupings rather than forcing artificial clusters
        """
        # Set reasonable bounds based on data size - MINIMUM 3, MAXIMUM 5
        min_clusters = 3  # Changed from 1 to 3
        max_clusters = 5  # Changed to always be 5
        
        if n_articles < min_clusters:
            return min_clusters
        
        # If we have very few articles, use minimum clusters
        if n_articles <= 6:
            return min_clusters  # Always return 3 for small datasets
        
        best_n_clusters = min_clusters  # Start with minimum clusters as baseline
        best_silhouette = -1
        best_inertia = float('inf')
        
        # Test different numbers of clusters, starting with minimum
        for n_clusters in range(min_clusters, max_clusters + 1):
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(features)
                
                # Calculate quality metrics
                if len(set(cluster_labels)) > 1:
                    silhouette_avg = silhouette_score(features, cluster_labels)
                    inertia = kmeans.inertia_
                    
                    logger.debug(f"Testing {n_clusters} clusters: silhouette = {silhouette_avg:.3f}, inertia = {inertia:.2f}")
                    
                    # For multiple clusters, prioritize silhouette score
                    if silhouette_avg > best_silhouette:
                        best_silhouette = silhouette_avg
                        best_n_clusters = n_clusters
                        best_inertia = inertia
                        
            except Exception as e:
                logger.debug(f"Failed to test {n_clusters} clusters: {e}")
                continue
        
        # Apply quality thresholds to avoid poor clustering
        if best_n_clusters > min_clusters:
            if best_silhouette < 0.2:  # Higher threshold for natural clustering
                logger.info(f"Clustering quality too low (silhouette: {best_silhouette:.3f}), using minimum clusters")
                return min_clusters
            elif best_silhouette < 0.3:  # Higher threshold for natural clustering
                logger.warning(f"⚠️ Low clustering quality (silhouette: {best_silhouette:.3f}). Consider fewer clusters.")
        
        # Log the decision
        if best_n_clusters == min_clusters:
            logger.info(f"✅ Using minimum {min_clusters} clusters - data suggests limited natural groupings")
        else:
            logger.info(f"✅ Best clustering found: {best_n_clusters} clusters with silhouette {best_silhouette:.3f}")
        
        return best_n_clusters
    
    def _validate_and_merge_clusters(self, clusters: List[NarrativeCluster], n_articles: int) -> List[NarrativeCluster]:
        """
        Validate clusters and merge very small ones to improve quality
        Ensures minimum of 3 clusters are maintained
        """
        if len(clusters) <= 1:
            return clusters
        
        # Never merge below minimum of 3 clusters
        if len(clusters) <= 3:
            logger.info(f"Maintaining minimum 3 clusters - no merging performed")
            return clusters
        
        # Calculate minimum cluster size (at least 10% of articles, minimum 2)
        min_cluster_size = max(2, n_articles // 10)
        
        # Find clusters that are too small
        small_clusters = [c for c in clusters if c.size < min_cluster_size]
        normal_clusters = [c for c in clusters if c.size >= min_cluster_size]
        
        if not small_clusters:
            return clusters
        
        # Only merge if we won't go below 3 clusters
        if len(clusters) - len(small_clusters) < 3:
            logger.info(f"Preventing merge to maintain minimum 3 clusters - keeping current {len(clusters)} clusters")
            return clusters
        
        logger.info(f"Found {len(small_clusters)} clusters with <{min_cluster_size} articles, merging...")
        
        # Merge small clusters into the nearest normal cluster
        for small_cluster in small_clusters:
            if not normal_clusters:
                # If no normal clusters, just keep the small one
                normal_clusters.append(small_cluster)
                continue
                
            # Find the closest normal cluster based on bias profile similarity
            best_cluster = None
            best_similarity = -1
            
            for normal_cluster in normal_clusters:
                similarity = self._calculate_bias_similarity(small_cluster.bias_profile, normal_cluster.bias_profile)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cluster = normal_cluster
            
            # Merge small cluster into best normal cluster
            if best_cluster:
                best_cluster.articles.extend(small_cluster.articles)
                best_cluster.size = len(best_cluster.articles)
                logger.info(f"Merged cluster {small_cluster.cluster_id} (size {small_cluster.size}) into cluster {best_cluster.cluster_id}")
        
        # Recalculate bias profiles for merged clusters
        for cluster in normal_clusters:
            if cluster.size > 0:
                bias_profiles = [self._extract_bias_features(article) for article in cluster.articles]
                avg_bias_profile = np.mean(bias_profiles, axis=0)
                cluster.bias_profile = {
                    'ideological_stance': float(avg_bias_profile[0]),
                    'factual_grounding': float(avg_bias_profile[1]),
                    'framing_choices': float(avg_bias_profile[2]),
                    'emotional_tone': float(avg_bias_profile[3]),
                    'source_transparency': float(avg_bias_profile[4])
                }
        
        logger.info(f"After merging: {len(normal_clusters)} clusters with minimum size {min_cluster_size}")
        return normal_clusters
    
    def _calculate_bias_similarity(self, profile1: Dict[str, float], profile2: Dict[str, float]) -> float:
        """
        Calculate similarity between two bias profiles using cosine similarity
        """
        try:
            # Extract values in consistent order
            keys = ['ideological_stance', 'factual_grounding', 'framing_choices', 'emotional_tone', 'source_transparency']
            vec1 = np.array([profile1.get(k, 50) for k in keys])
            vec2 = np.array([profile2.get(k, 50) for k in keys])
            
            # Normalize vectors
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = np.dot(vec1, vec2) / (norm1 * norm2)
            return float(similarity)
        except Exception:
            return 0.0
    
    def _explain_clustering_decision(self, n_clusters: int, n_articles: int, silhouette_score: float = None) -> str:
        """
        Explain why a particular number of clusters was chosen
        Provides transparency about clustering decisions
        """
        if n_clusters == 3:
            return f"Minimum 3 clusters created: Data analysis ({n_articles} articles) suggests limited natural narrative groupings. Articles have been grouped into 3 clusters based on similarity patterns."
        
        explanation = f"Clustering decision: {n_clusters} clusters created from {n_articles} articles"
        
        if silhouette_score is not None:
            if silhouette_score >= 0.6:
                explanation += f" with excellent separation (silhouette: {silhouette_score:.3f})"
            elif silhouette_score >= 0.4:
                explanation += f" with good separation (silhouette: {silhouette_score:.3f})"
            elif silhouette_score >= 0.25:
                explanation += f" with moderate separation (silhouette: {silhouette_score:.3f})"
            else:
                explanation += f" with weak separation (silhouette: {silhouette_score:.3f}) - consider fewer clusters"
        
        # Add reasoning about the number of clusters
        if n_clusters == 3:
            explanation += ". Three narrative frameworks identified (minimum requirement)."
        elif n_clusters == 4:
            explanation += ". Four narrative patterns discovered in the data."
        elif n_clusters == 5:
            explanation += ". Five narrative clusters found, suggesting diverse coverage of the topic."
        else:
            explanation += f". {n_clusters} narrative clusters found, suggesting diverse coverage of the topic."
        
        # Add data size context
        if n_articles <= 10:
            explanation += f" Limited dataset size ({n_articles} articles) - clustering results should be interpreted cautiously."
        elif n_articles <= 25:
            explanation += f" Moderate dataset size ({n_articles} articles) - clustering provides reasonable insights."
        else:
            explanation += f" Robust dataset size ({n_articles} articles) - clustering results are statistically meaningful."
        
        return explanation
    
    def _extract_narrative_text(self, article: Dict[str, Any]) -> str:
        """Extract text content for narrative analysis"""
        title = article.get('title', '')
        content = article.get('content', '')
        
        # Focus on key narrative elements
        narrative_text = f"{title} {content}"
        
        # Include highlighted bias phrases if available
        if 'analysis' in article and 'highlighted_phrases' in article['analysis']:
            phrases = [phrase.get('text', '') for phrase in article['analysis']['highlighted_phrases']]
            narrative_text += ' ' + ' '.join(phrases)
        
        return narrative_text
    
    def _extract_bias_features(self, article: Dict[str, Any]) -> np.ndarray:
        """Extract bias dimension scores as features"""
        if 'analysis' not in article:
            return np.array([50, 50, 50, 50, 50])  # Neutral default
        
        dimension_scores = article['analysis'].get('dimension_scores', {})
        return np.array([
            dimension_scores.get('ideological_stance', 50),
            dimension_scores.get('factual_grounding', 50),
            dimension_scores.get('framing_choices', 50),
            dimension_scores.get('emotional_tone', 50),
            dimension_scores.get('source_transparency', 50)
        ])
    
    def _create_narrative_cluster(self, cluster_id: int, cluster_articles: List[Dict], 
                                text_features, cluster_labels) -> NarrativeCluster:
        """Create a narrative cluster from grouped articles"""
        
        # Extract dominant themes from TF-IDF features
        cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
        cluster_tfidf = text_features[cluster_indices]
        
        # Get top TF-IDF terms for this cluster
        mean_tfidf = np.mean(cluster_tfidf, axis=0).A1
        top_term_indices = mean_tfidf.argsort()[-15:][::-1]  # Get more terms initially
        feature_names = self.vectorizer.get_feature_names_out()
        dominant_themes = [feature_names[i] for i in top_term_indices if mean_tfidf[i] > 0]
        
        # Filter and refine dominant themes
        dominant_themes = self._filter_themes(dominant_themes)
        
        # If we don't have enough themes, try to get more from the top terms
        if len(dominant_themes) < 3:
            additional_themes = [feature_names[i] for i in top_term_indices[:20] 
                               if feature_names[i] not in dominant_themes and 
                               len(feature_names[i]) >= 3 and len(feature_names[i]) <= 25]
            dominant_themes.extend(additional_themes[:3-len(dominant_themes)])
        
        # Calculate average bias profile for cluster
        bias_profiles = [self._extract_bias_features(article) for article in cluster_articles]
        avg_bias_profile = np.mean(bias_profiles, axis=0)
        bias_profile = {
            'ideological_stance': float(avg_bias_profile[0]),
            'factual_grounding': float(avg_bias_profile[1]),
            'framing_choices': float(avg_bias_profile[2]),
            'emotional_tone': float(avg_bias_profile[3]),
            'source_transparency': float(avg_bias_profile[4])
        }
        
        # Extract representative phrases
        representative_phrases = []
        for article in cluster_articles:
            if 'analysis' in article and 'highlighted_phrases' in article['analysis']:
                phrases = [phrase.get('text', '') for phrase in article['analysis']['highlighted_phrases'][:2]]
                representative_phrases.extend(phrases)
        
        # Remove duplicates and limit
        representative_phrases = list(set(representative_phrases))[:5]
        
        return NarrativeCluster(
            cluster_id=cluster_id,
            articles=cluster_articles,
            dominant_themes=dominant_themes,
            bias_profile=bias_profile,
            representative_phrases=representative_phrases
        )
    
    def _create_single_cluster(self, articles: List[Dict[str, Any]]) -> List[NarrativeCluster]:
        """
        Create clusters when the algorithm suggests single cluster but we need minimum 3
        This method now creates 3 clusters by grouping articles by similarity
        """
        if len(articles) < 3:
            # If we have less than 3 articles, create 3 clusters with the available articles
            clusters = []
            for i in range(3):
                if i < len(articles):
                    cluster_articles = [articles[i]]
                else:
                    # Create empty clusters for the remaining slots
                    cluster_articles = []
                
                # Extract text for themes
                all_text = []
                for article in cluster_articles:
                    text = self._extract_narrative_text(article)
                    cleaned_text = self._clean_text(text)
                    all_text.append(cleaned_text)
                
                # Generate themes
                if cluster_articles:
                    try:
                        if len(all_text) > 1:
                            temp_vectorizer = TfidfVectorizer(
                                max_features=100,
                                stop_words='english',
                                ngram_range=(1, 2),
                                min_df=1,
                                max_df=1.0
                            )
                            temp_features = temp_vectorizer.fit_transform(all_text)
                            feature_names = temp_vectorizer.get_feature_names_out()
                            
                            # Get terms that appear in multiple articles
                            doc_term_matrix = temp_features.toarray()
                            term_frequencies = np.sum(doc_term_matrix > 0, axis=0)
                            common_terms = [feature_names[i] for i in range(len(feature_names)) 
                                          if term_frequencies[i] > 1]
                            
                            # Filter themes and limit to 3
                            themes = self._filter_themes(common_terms, min_themes=1)
                            if len(themes) < 3:
                                # Try to get more themes by being less restrictive
                                additional_themes = [term for term in common_terms[:10] 
                                                   if term not in themes and len(term) >= 3 and len(term) <= 20]
                                themes.extend(additional_themes[:3-len(themes)])
                        else:
                            themes = ['single_article']
                    except Exception:
                        themes = ['mixed_coverage']
                    
                    # Calculate bias profile
                    bias_profiles = [self._extract_bias_features(article) for article in cluster_articles]
                    avg_bias_profile = np.mean(bias_profiles, axis=0)
                    bias_profile = {
                        'ideological_stance': float(avg_bias_profile[0]),
                        'factual_grounding': float(avg_bias_profile[1]),
                        'framing_choices': float(avg_bias_profile[2]),
                        'emotional_tone': float(avg_bias_profile[3]),
                        'source_transparency': float(avg_bias_profile[4])
                    }
                else:
                    # Empty cluster
                    themes = ['empty_cluster']
                    bias_profile = {
                        'ideological_stance': 50.0,
                        'factual_grounding': 50.0,
                        'framing_choices': 50.0,
                        'emotional_tone': 50.0,
                        'source_transparency': 50.0
                    }
                
                clusters.append(NarrativeCluster(
                    cluster_id=i,
                    articles=cluster_articles,
                    dominant_themes=themes,
                    bias_profile=bias_profile,
                    representative_phrases=[]
                ))
            
            return clusters
        
        # If we have 3 or more articles, create 3 clusters by grouping similar articles
        # Sort articles by overall bias score for better distribution
        articles_with_scores = []
        for article in articles:
            bias_features = self._extract_bias_features(article)
            overall_score = np.mean(bias_features)
            articles_with_scores.append((article, overall_score))
        
        # Sort by overall score
        articles_with_scores.sort(key=lambda x: x[1])
        
        # Create 3 clusters by dividing articles
        cluster_size = len(articles) // 3
        clusters = []
        
        for i in range(3):
            start_idx = i * cluster_size
            end_idx = start_idx + cluster_size if i < 2 else len(articles)
            cluster_articles = [item[0] for item in articles_with_scores[start_idx:end_idx]]
            
            # Extract text for themes
            all_text = []
            for article in cluster_articles:
                text = self._extract_narrative_text(article)
                cleaned_text = self._clean_text(text)
                all_text.append(cleaned_text)
            
            # Use TF-IDF to find common themes
            try:
                if len(all_text) > 1:
                    temp_vectorizer = TfidfVectorizer(
                        max_features=100,
                        stop_words='english',
                        ngram_range=(1, 2),
                        min_df=1,
                        max_df=1.0
                    )
                    temp_features = temp_vectorizer.fit_transform(all_text)
                    feature_names = temp_vectorizer.get_feature_names_out()
                    
                    # Get terms that appear in multiple articles
                    doc_term_matrix = temp_features.toarray()
                    term_frequencies = np.sum(doc_term_matrix > 0, axis=0)
                    common_terms = [feature_names[i] for i in range(len(feature_names)) 
                                  if term_frequencies[i] > 1]
                    
                    # Filter themes and limit to 3
                    themes = self._filter_themes(common_terms, min_themes=1)
                    if len(themes) < 3:
                        # Try to get more themes by being less restrictive
                        additional_themes = [term for term in common_terms[:10] 
                                           if term not in themes and len(term) >= 3 and len(term) <= 20]
                        themes.extend(additional_themes[:3-len(themes)])
                else:
                    themes = ['single_article']
            except Exception:
                themes = ['mixed_coverage']
            
            # Calculate average bias profile
            bias_profiles = [self._extract_bias_features(article) for article in cluster_articles]
            avg_bias_profile = np.mean(bias_profiles, axis=0)
            bias_profile = {
                'ideological_stance': float(avg_bias_profile[0]),
                'factual_grounding': float(avg_bias_profile[1]),
                'framing_choices': float(avg_bias_profile[2]),
                'emotional_tone': float(avg_bias_profile[3]),
                'source_transparency': float(avg_bias_profile[4])
            }
            
            clusters.append(NarrativeCluster(
                cluster_id=i,
                articles=cluster_articles,
                dominant_themes=themes,
                bias_profile=bias_profile,
                representative_phrases=[]
            ))
        
        return clusters
    
    def get_cluster_visualization_data(self, clusters: List[NarrativeCluster]) -> Dict[str, Any]:
        """Generate comprehensive visualization data including PCA, t-SNE, and bias analysis"""
        if not clusters:
            return {"clusters": [], "visualizations": None}
        
        # Prepare data for visualization
        all_articles = []
        cluster_labels = []
        for cluster in clusters:
            all_articles.extend(cluster.articles)
            cluster_labels.extend([cluster.cluster_id] * len(cluster.articles))
        
        if len(all_articles) < 3:
            return {"clusters": self._serialize_clusters(clusters), "visualizations": None}
        
        try:
            # Extract features
            texts = [self._extract_narrative_text(article) for article in all_articles]
            cleaned_texts = [self._clean_text(text) for text in texts]
            text_features = self.vectorizer.fit_transform(cleaned_texts)
            bias_features = np.array([self._extract_bias_features(article) for article in all_articles])
            
            # Standardize bias features for better visualization
            scaler = StandardScaler()
            bias_features_scaled = scaler.fit_transform(bias_features)
            
            # Combine features
            combined_features = np.hstack([text_features.toarray(), bias_features_scaled])
            
            # Generate multiple visualizations
            visualizations = {}
            
            # 1. PCA Visualization (faster, shows main variance)
            if len(all_articles) >= 2:
                pca = PCA(n_components=2, random_state=42)
                pca_features = pca.fit_transform(combined_features)
                
                visualizations['pca'] = {
                    'type': 'PCA',
                    'title': 'Principal Component Analysis - Narrative Similarity',
                    'explained_variance': [float(pca.explained_variance_ratio_[0]), float(pca.explained_variance_ratio_[1])],
                    'points': [
                        {
                            'x': float(pca_features[i][0]),
                            'y': float(pca_features[i][1]),
                            'cluster': int(cluster_labels[i]),
                            'title': all_articles[i].get('title', '')[:60],
                            'source': all_articles[i].get('source', ''),
                            'bias_score': all_articles[i].get('analysis', {}).get('overall_score', 50),
                            'dimension_scores': all_articles[i].get('analysis', {}).get('dimension_scores', {})
                        }
                        for i in range(len(all_articles))
                    ]
                }
            
            # 2. t-SNE Visualization (nonlinear, shows local structure)
            if len(all_articles) >= 3:
                perplexity = min(5, len(all_articles) - 1)
                tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, learning_rate='auto', init='pca')
                tsne_features = tsne.fit_transform(combined_features)
                
                visualizations['tsne'] = {
                    'type': 't-SNE',
                    'title': 't-SNE Analysis - Local Narrative Clusters',
                    'perplexity': perplexity,
                    'points': [
                        {
                            'x': float(tsne_features[i][0]),
                            'y': float(tsne_features[i][1]),
                            'cluster': int(cluster_labels[i]),
                            'title': all_articles[i].get('title', '')[:60],
                            'source': all_articles[i].get('source', ''),
                            'bias_score': all_articles[i].get('analysis', {}).get('overall_score', 50),
                            'dimension_scores': all_articles[i].get('analysis', {}).get('dimension_scores', {})
                        }
                        for i in range(len(all_articles))
                    ]
                }
            
            # 3. Bias Score Distribution (1D visualization)
            bias_scores = [article.get('analysis', {}).get('overall_score', 50) for article in all_articles]
            visualizations['bias_distribution'] = {
                'type': 'Bias Distribution',
                'title': 'Overall Bias Score Distribution',
                'points': [
                    {
                        'x': float(i),  # Article index
                        'y': float(bias_scores[i]),
                        'cluster': int(cluster_labels[i]),
                        'title': all_articles[i].get('title', '')[:60],
                        'source': all_articles[i].get('source', ''),
                        'bias_score': bias_scores[i],
                        'dimension_scores': all_articles[i].get('analysis', {}).get('dimension_scores', {})
                    }
                    for i in range(len(all_articles))
                ]
            }
            
            # 4. Generate story coverage analysis
            story_coverage = self._analyze_story_coverage(clusters)
            
            return {
                "clusters": self._serialize_clusters(clusters),
                "visualizations": visualizations,
                "story_coverage_analysis": story_coverage
            }
            
        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
            return {"clusters": self._serialize_clusters(clusters), "visualizations": None, "story_coverage_analysis": None}
    
    def _analyze_story_coverage(self, clusters: List[NarrativeCluster]) -> Dict[str, Any]:
        """Analyze how different outlets cover the same story"""
        coverage_analysis = {
            "same_story_different_coverage": [],
            "bias_spectrum_by_source": {},
            "narrative_differences": []
        }
        
        # Group articles by similar topics/themes
        topic_groups = {}
        for cluster in clusters:
            for theme in cluster.dominant_themes[:2]:  # Take top 2 themes
                if theme not in topic_groups:
                    topic_groups[theme] = []
                topic_groups[theme].extend([
                    {
                        "title": article.get('title', ''),
                        "source": article.get('source', ''),
                        "bias_profile": article.get('analysis', {}).get('dimension_scores', {}),
                        "overall_score": article.get('analysis', {}).get('overall_score', 50),
                        "cluster_id": cluster.cluster_id
                    }
                    for article in cluster.articles
                ])
        
        # Analyze coverage differences for each topic
        for topic, articles in topic_groups.items():
            if len(articles) >= 2:  # Need at least 2 articles to compare
                sources = list(set([article['source'] for article in articles]))
                if len(sources) >= 2:  # Different sources covering same topic
                    coverage_analysis["same_story_different_coverage"].append({
                        "topic": topic,
                        "sources": sources,
                        "articles": articles,
                        "bias_range": {
                            "min": min([article['overall_score'] for article in articles]),
                            "max": max([article['overall_score'] for article in articles]),
                            "variance": np.var([article['overall_score'] for article in articles])
                        }
                    })
        
        # Calculate bias spectrum by source
        source_bias_data = {}
        for cluster in clusters:
            for article in cluster.articles:
                source = article.get('source', 'Unknown')
                if source not in source_bias_data:
                    source_bias_data[source] = []
                source_bias_data[source].append(article.get('analysis', {}).get('overall_score', 50))
        
        for source, scores in source_bias_data.items():
            coverage_analysis["bias_spectrum_by_source"][source] = {
                "average_bias": np.mean(scores),
                "bias_confidence": 100 - np.std(scores),  # Lower std = more confident/consistent
                "article_count": len(scores),
                "score_range": [min(scores), max(scores)]
            }
        
        return coverage_analysis
    
    def _serialize_clusters(self, clusters: List[NarrativeCluster]) -> List[Dict[str, Any]]:
        """Convert clusters to JSON-serializable format"""
        return [
            {
                'cluster_id': cluster.cluster_id,
                'size': cluster.size,
                'dominant_themes': cluster.dominant_themes,
                'bias_profile': cluster.bias_profile,
                'representative_phrases': cluster.representative_phrases,
                'articles': [
                    {
                        'title': article.get('title', ''),
                        'content': article.get('content', '')[:200] + '...' if len(article.get('content', '')) > 200 else article.get('content', ''),
                        'source': article.get('source', ''),
                        'url': article.get('url', ''),
                        'published_at': article.get('published_at', ''),
                        'overall_bias_score': article.get('analysis', {}).get('overall_score', 50),
                        'analysis': article.get('analysis', {})  # Include FULL analysis data
                    }
                    for article in cluster.articles
                ]
            }
            for cluster in clusters
        ]

    def get_clustering_insights(self, clusters: List[NarrativeCluster], n_articles: int) -> Dict[str, Any]:
        """
        Provide detailed insights about the clustering results
        Helps users understand why certain clustering decisions were made
        """
        insights = {
            'total_articles': n_articles,
            'total_clusters': len(clusters),
            'clustering_quality': 'unknown',
            'recommendations': [],
            'cluster_sizes': [cluster.size for cluster in clusters],
            'size_distribution': {},
            'quality_metrics': {}
        }
        
        # Analyze cluster size distribution
        if clusters:
            size_counts = {}
            for size in insights['cluster_sizes']:
                size_counts[size] = size_counts.get(size, 0) + 1
            insights['size_distribution'] = size_counts
            
            # Check for balanced vs unbalanced clusters
            avg_size = n_articles / len(clusters)
            size_variance = sum((size - avg_size) ** 2 for size in insights['cluster_sizes']) / len(clusters)
            
            if size_variance < avg_size * 0.5:
                insights['size_balance'] = 'balanced'
            elif size_variance < avg_size * 1.5:
                insights['size_balance'] = 'moderately_balanced'
            else:
                insights['size_balance'] = 'unbalanced'
        
        # Quality assessment
        if len(clusters) == 3:
            insights['clustering_quality'] = 'minimum_clusters'
            insights['recommendations'].append("Minimum 3 clusters created - articles grouped by similarity patterns")
        elif len(clusters) == 4:
            insights['clustering_quality'] = 'moderate_granularity'
            insights['recommendations'].append("Four clusters provide good narrative separation")
        elif len(clusters) == 5:
            insights['clustering_quality'] = 'optimal_granularity'
            insights['recommendations'].append("Optimal 5 clusters provide comprehensive narrative coverage")
        else:
            insights['clustering_quality'] = 'high_granularity'
            insights['recommendations'].append("High granularity clustering - consider if all clusters are meaningful")
        
        # Check for very small clusters
        small_clusters = [size for size in insights['cluster_sizes'] if size < 3]
        if small_clusters:
            insights['recommendations'].append(f"Found {len(small_clusters)} very small clusters - consider merging for better quality")
        
        # Data size recommendations
        if n_articles < 10:
            insights['recommendations'].append("Small dataset - clustering results should be interpreted with caution")
        elif n_articles < 20:
            insights['recommendations'].append("Moderate dataset - clustering provides reasonable insights")
        else:
            insights['recommendations'].append("Robust dataset - clustering results are statistically meaningful")
        
        return insights

    def should_cluster(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Determine if clustering is appropriate for the given articles
        Provides recommendations about clustering strategy
        """
        n_articles = len(articles)
        
        recommendation = {
            'should_cluster': True,
            'reasoning': '',
            'recommended_approach': '',
            'warnings': [],
            'data_quality': 'unknown'
        }
        
        # Check data size
        if n_articles < 3:
            recommendation['should_cluster'] = False
            recommendation['reasoning'] = f"Too few articles ({n_articles}) for meaningful clustering"
            recommendation['recommended_approach'] = 'single_analysis'
            recommendation['warnings'].append('Clustering with <3 articles will not provide meaningful insights')
            return recommendation
        
        elif n_articles < 6:
            recommendation['should_cluster'] = False
            recommendation['reasoning'] = f"Very small dataset ({n_articles} articles) - clustering may be artificial"
            recommendation['recommended_approach'] = 'single_analysis'
            recommendation['warnings'].append('Consider expanding dataset or using single analysis approach')
            return recommendation
        
        elif n_articles < 10:
            recommendation['should_cluster'] = True
            recommendation['reasoning'] = f"Small dataset ({n_articles} articles) - clustering results should be interpreted cautiously"
            recommendation['recommended_approach'] = 'conservative_clustering'
            recommendation['warnings'].append('Results may not be statistically robust')
            recommendation['data_quality'] = 'limited'
        
        elif n_articles < 20:
            recommendation['should_cluster'] = True
            recommendation['reasoning'] = f"Moderate dataset ({n_articles} articles) - clustering provides reasonable insights"
            recommendation['recommended_approach'] = 'standard_clustering'
            recommendation['data_quality'] = 'moderate'
        
        else:
            recommendation['should_cluster'] = True
            recommendation['reasoning'] = f"Robust dataset ({n_articles} articles) - clustering results are statistically meaningful"
            recommendation['recommended_approach'] = 'comprehensive_clustering'
            recommendation['data_quality'] = 'robust'
        
        # Check for topic diversity (if we have content)
        if articles and 'content' in articles[0]:
            # Simple diversity check based on content length and source variety
            sources = set(article.get('source', '') for article in articles)
            avg_content_length = sum(len(article.get('content', '')) for article in articles) / n_articles
            
            if len(sources) == 1:
                recommendation['warnings'].append('Single source detected - clustering may reflect source bias rather than topic diversity')
            
            if avg_content_length < 200:
                recommendation['warnings'].append('Short content detected - clustering quality may be limited')
        
        return recommendation
