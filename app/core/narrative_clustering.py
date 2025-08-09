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
from typing import List, Dict, Any, Tuple
import structlog

logger = structlog.get_logger()


class NarrativeCluster:
    """Represents a cluster of articles with similar narrative framing"""
    def __init__(self, cluster_id: int, articles: List[Dict], dominant_themes: List[str], 
                 bias_profile: Dict[str, float], representative_phrases: List[str]):
        self.cluster_id = cluster_id
        self.articles = articles
        self.dominant_themes = dominant_themes
        self.bias_profile = bias_profile
        self.representative_phrases = representative_phrases
        self.size = len(articles)


class NarrativeClusteringEngine:
    """Clusters articles by narrative similarity and bias patterns"""
    
    def __init__(self, max_clusters: int = 5):
        self.max_clusters = max_clusters
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 3),  # Include phrases
            min_df=2,
            max_df=0.8
        )
        
    def cluster_narratives(self, articles_with_analysis: List[Dict[str, Any]]) -> List[NarrativeCluster]:
        """
        Cluster articles based on narrative similarity and bias patterns
        """
        if len(articles_with_analysis) < 3:
            logger.warning("Too few articles for meaningful clustering")
            return self._create_single_cluster(articles_with_analysis)
        
        try:
            # Extract text features
            texts = [self._extract_narrative_text(article) for article in articles_with_analysis]
            text_features = self.vectorizer.fit_transform(texts)
            
            # Extract bias features
            bias_features = np.array([self._extract_bias_features(article) for article in articles_with_analysis])
            
            # Combine text and bias features
            combined_features = np.hstack([
                text_features.toarray(),
                bias_features
            ])
            
            # Determine optimal number of clusters
            n_clusters = min(self.max_clusters, len(articles_with_analysis) // 2, 
                           max(2, len(articles_with_analysis) // 3))
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(combined_features)
            
            # Calculate silhouette score for cluster quality
            if len(set(cluster_labels)) > 1:
                silhouette_avg = silhouette_score(combined_features, cluster_labels)
                logger.info(f"Clustering silhouette score: {silhouette_avg:.3f}")
            
            # Create narrative clusters
            clusters = []
            for cluster_id in range(n_clusters):
                cluster_articles = [articles_with_analysis[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
                if cluster_articles:
                    cluster = self._create_narrative_cluster(cluster_id, cluster_articles, text_features, cluster_labels)
                    clusters.append(cluster)
            
            logger.info(f"Created {len(clusters)} narrative clusters")
            return clusters
            
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            return self._create_single_cluster(articles_with_analysis)
    
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
        top_term_indices = mean_tfidf.argsort()[-10:][::-1]
        feature_names = self.vectorizer.get_feature_names_out()
        dominant_themes = [feature_names[i] for i in top_term_indices if mean_tfidf[i] > 0]
        
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
            dominant_themes=dominant_themes[:5],
            bias_profile=bias_profile,
            representative_phrases=representative_phrases
        )
    
    def _create_single_cluster(self, articles: List[Dict[str, Any]]) -> List[NarrativeCluster]:
        """Create a single cluster when clustering fails or too few articles"""
        if not articles:
            return []
        
        # Calculate average bias profile
        bias_profiles = [self._extract_bias_features(article) for article in articles]
        avg_bias_profile = np.mean(bias_profiles, axis=0)
        bias_profile = {
            'ideological_stance': float(avg_bias_profile[0]),
            'factual_grounding': float(avg_bias_profile[1]),
            'framing_choices': float(avg_bias_profile[2]),
            'emotional_tone': float(avg_bias_profile[3]),
            'source_transparency': float(avg_bias_profile[4])
        }
        
        return [NarrativeCluster(
            cluster_id=0,
            articles=articles,
            dominant_themes=['mixed_coverage'],
            bias_profile=bias_profile,
            representative_phrases=[]
        )]
    
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
            text_features = self.vectorizer.fit_transform(texts)
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
