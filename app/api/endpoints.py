"""
FastAPI endpoints for the Advanced Bias Detection Engine
Real-time news analysis with narrative clustering
"""
import asyncio
import time
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import structlog

from app.core.bias_detector import BiasDetectionEngine
from app.core.real_bias_analyzer import RealBiasAnalyzer
from app.core.news_fetcher import NewsAPIFetcher, RSSNewsFetcher
from app.core.narrative_clustering import NarrativeClusteringEngine
from app.models.schemas import (
    ArticleInput, BiasAnalysisOutput, BatchAnalysisRequest,
    BatchAnalysisResponse, NewsAnalysisRequest, NewsAnalysisResponse,
    NarrativeClusterOutput
)
from config.settings import settings

logger = structlog.get_logger()

# Initialize FastAPI app
app = FastAPI(
    title="Advanced Bias Detection Engine",
    description="Real-time news bias analysis with narrative clustering and explainable AI",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize core engines
bias_detector = BiasDetectionEngine()
real_bias_analyzer = RealBiasAnalyzer()  # NEW: Real NLP-based analysis
news_fetcher = NewsAPIFetcher()  # Add API key from settings if available
rss_fetcher = RSSNewsFetcher()
clustering_engine = NarrativeClusteringEngine()

# Mount static files for web interface
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
async def serve_web_interface():
    """Serve the main web interface with advanced bias detection"""
    return FileResponse("app/static/index.html")

@app.get("/api")
async def api_info():
    """API information endpoint"""
    return {
        "message": "Advanced Bias Detection Engine API",
        "version": "2.0.0",
        "features": [
            "Real-time news analysis",
            "Sub-200ms bias scoring",
            "Narrative clustering",
            "Explainable AI with phrase highlighting",
            "Confidence intervals",
            "Multi-source data ingestion"
        ],
        "status": "operational",
        "web_interface": "/",
        "docs": "/docs",
        "timestamp": time.time()
    }


@app.get("/health")
async def health_check():
    """Detailed health check with system status"""
    return {
        "status": "healthy",
        "services": {
            "bias_detector": "operational",
            "news_fetcher": "operational",
            "clustering_engine": "operational",
            "api": "operational"
        },
        "performance": {
            "cache_enabled": True,
            "max_concurrent_requests": settings.max_concurrent_requests,
            "target_response_time_ms": settings.response_timeout_ms
        },
        "features": {
            "narrative_clustering": True,
            "explainable_ai": True,
            "confidence_intervals": True,
            "real_time_news": True
        }
    }


@app.post("/analyze/article", response_model=BiasAnalysisOutput)
async def analyze_article(article: ArticleInput):
    """
    Analyze a single article for bias with explainable AI and confidence intervals
    """
    start_time = time.time()

    try:
        if not article.content or len(article.content.strip()) < 50:
            raise HTTPException(
                status_code=400,
                detail="Article content must be at least 50 characters"
            )

        # Use real NLP-based bias analysis
        analysis_result = await real_bias_analyzer.analyze_article(
            title=article.title,
            content=article.content,
            source=article.source
        )

        processing_time = (time.time() - start_time) * 1000
        logger.info("Article analyzed with explainable AI",
                   article_id=analysis_result.get("article_id"),
                   processing_time_ms=processing_time,
                   overall_score=analysis_result.get("overall_score"),
                   num_highlighted_phrases=len(analysis_result.get("highlighted_phrases", [])))

        if processing_time > settings.response_timeout_ms:
            logger.warning("Response time exceeded target",
                          processing_time_ms=processing_time,
                          target_ms=settings.response_timeout_ms)

        return BiasAnalysisOutput(**analysis_result)

    except Exception as e:
        logger.error("Article analysis failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/analyze/batch", response_model=BatchAnalysisResponse)
async def analyze_batch(request: BatchAnalysisRequest, background_tasks: BackgroundTasks):
    """
    Analyze multiple articles in a batch with performance optimization
    """
    start_time = time.time()

    try:
        if len(request.articles) > 20:
            raise HTTPException(
                status_code=400,
                detail="Batch size limited to 20 articles for optimal performance"
            )

        analysis_tasks = [
            bias_detector.analyze_article(
                title=article.title,
                content=article.content,
                source=article.source
            )
            for article in request.articles
        ]

        results = await asyncio.gather(*analysis_tasks)
        successful_analyses = [BiasAnalysisOutput(**res) for res in results if res]

        total_processing_time = (time.time() - start_time) * 1000
        batch_id = hashlib.md5(str(time.time()).encode()).hexdigest()

        logger.info("Batch analysis completed",
                   batch_id=batch_id,
                   num_articles=len(successful_analyses),
                   total_processing_time_ms=total_processing_time,
                   avg_processing_time_ms=total_processing_time / len(successful_analyses) if successful_analyses else 0)

        return BatchAnalysisResponse(
            analyses=successful_analyses,
            batch_id=batch_id,
            total_processing_time_ms=total_processing_time,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error("Batch analysis failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")


@app.post("/analyze/news", response_model=NewsAnalysisResponse)
async def analyze_current_news(request: NewsAnalysisRequest):
    """
    Fetch and analyze current news articles with narrative clustering
    Real-time bias detection pipeline
    """
    start_time = time.time()

    try:
        # Fetch articles from news sources
        logger.info("Fetching current news articles", topic=request.topic, max_articles=request.max_articles)
        
        if request.topic:
            # ALWAYS use topic-specific articles when topic is provided
            articles = await news_fetcher.fetch_trending_topic(
                query=request.topic,
                sources=request.sources
            )
            logger.info(f"Using topic-specific articles for: {request.topic}")
        else:
            # Use RSS feeds only when no topic specified
            articles = await rss_fetcher.fetch_diverse_coverage(max_articles=request.max_articles)
            logger.info("Using RSS feeds for general coverage")
        
        if not articles:
            raise HTTPException(status_code=404, detail="No articles found for the specified topic")

        # Analyze each article for bias using REAL NLP techniques
        logger.info(f"Analyzing {len(articles)} articles for bias using real NLP methods")
        analysis_tasks = [
            real_bias_analyzer.analyze_article(
                title=article.title,
                content=article.content,
                source=article.source
            )
            for article in articles
        ]
        
        analyses = await asyncio.gather(*analysis_tasks)
        logger.info(f"Real NLP analysis completed for {len(analyses)} articles")
        
        # Combine articles with their analyses
        articles_with_analysis = []
        for article, analysis in zip(articles, analyses):
            articles_with_analysis.append({
                'title': article.title,
                'content': article.content,
                'source': article.source,
                'url': article.url,
                'published_at': article.published_at,
                'analysis': analysis
            })

        # Perform narrative clustering
        logger.info("Performing narrative clustering analysis")
        clusters = clustering_engine.cluster_narratives(articles_with_analysis)
        
        # Generate visualization data
        cluster_viz_data = clustering_engine.get_cluster_visualization_data(clusters)

        total_processing_time = (time.time() - start_time) * 1000
        
        # Convert clusters to response format
        cluster_outputs = [
            NarrativeClusterOutput(
                cluster_id=cluster['cluster_id'],
                size=cluster['size'],
                dominant_themes=cluster['dominant_themes'],
                bias_profile=cluster['bias_profile'],
                representative_phrases=cluster['representative_phrases'],
                articles=cluster['articles']
            )
            for cluster in cluster_viz_data['clusters']
        ]

        logger.info("News analysis pipeline completed",
                   articles_analyzed=len(articles),
                   clusters_found=len(clusters),
                   total_processing_time_ms=total_processing_time)

        return NewsAnalysisResponse(
            topic=request.topic or "Current News",
            articles_analyzed=len(articles),
            processing_time_ms=total_processing_time,
            processing_time_per_article_ms=total_processing_time / len(articles) if articles else 0,
            narrative_clusters=cluster_outputs,
            cluster_visualizations=cluster_viz_data.get('visualizations'),
            story_coverage_analysis=cluster_viz_data.get('story_coverage_analysis'),
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error("News analysis failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"News analysis failed: {str(e)}")


@app.get("/demo/sample-articles")
async def get_sample_articles():
    """
    Provides sample articles for demo purposes with diverse bias perspectives
    """
    sample_articles = [
        {
            "title": "Breaking: Congressional Leaders Clash Over Infrastructure Bill",
            "content": "In a heated exchange that shocked Washington insiders, progressive lawmakers slammed moderate Democrats for their 'devastating betrayal' of working families. The controversial infrastructure package, described by critics as a 'radical socialist agenda' and by supporters as 'groundbreaking reform,' has exposed deep fractures within the party. Sources familiar with the negotiations claim that backroom deals and special interests have corrupted the process, while others insist the bill represents historic progress for America's future.",
            "source": "Political News Daily"
        },
        {
            "title": "Infrastructure Investment: A Balanced Approach to National Priorities",
            "content": "Congressional leaders from both parties are working to craft comprehensive infrastructure legislation that addresses America's critical needs. The proposed package includes investments in transportation, communications, and clean energy infrastructure. While negotiations continue, lawmakers emphasize their commitment to finding common ground that serves the American people. Industry experts praise the bill's focus on job creation and economic competitiveness.",
            "source": "National Policy Review"
        },
        {
            "title": "Taxpayers Beware: Another Massive Government Spending Spree",
            "content": "Once again, Washington politicians are pushing through another bloated spending bill that will burden hardworking taxpayers for generations. This so-called 'infrastructure' package is packed with liberal pet projects and Green New Deal fantasies that have nothing to do with fixing roads and bridges. Conservative analysts warn that this reckless spending will fuel inflation and expand government control over the economy.",
            "source": "Conservative Tribune"
        }
    ]
    return sample_articles


@app.get("/demo/current-news")
async def get_current_controversial_news():
    """
    Fetch and analyze current controversial news for demonstration
    """
    try:
        # Use demo articles that represent different bias perspectives
        articles = await rss_fetcher.fetch_diverse_coverage(max_articles=10)
        
        # Quick analysis for demo
        analyzed_articles = []
        for article in articles[:5]:  # Limit for demo speed
            analysis = await bias_detector.analyze_article(
                title=article.title,
                content=article.content,
                source=article.source
            )
            analyzed_articles.append({
                'title': article.title,
                'source': article.source,
                'bias_score': analysis['overall_score'],
                'key_phrases': [phrase.get('text', '') for phrase in analysis.get('highlighted_phrases', [])[:2]]
            })
        
        return {
            "message": "Current news analysis demo",
            "articles_analyzed": len(analyzed_articles),
            "articles": analyzed_articles,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error("Demo news fetch failed", error=str(e))
        return {
            "message": "Demo mode - using sample data",
            "articles_analyzed": 0,
            "articles": [],
            "error": str(e)
        }


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "status_code": 404}


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error", "status_code": 500}