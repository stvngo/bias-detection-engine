"""
Real-time news fetching for bias analysis pipeline
Pulls current controversial topics from multiple sources
"""
import asyncio
import aiohttp
import feedparser
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import hashlib
import structlog

logger = structlog.get_logger()


class NewsArticle:
    """Represents a news article for bias analysis"""
    def __init__(self, title: str, content: str, source: str, url: str, published_at: str):
        self.title = title
        self.content = content
        self.source = source
        self.url = url
        self.published_at = published_at
        self.id = hashlib.md5(f"{title}{source}".encode()).hexdigest()


class NewsAPIFetcher:
    """Fetches news from NewsAPI.org for current controversial topics"""
    
    def __init__(self, api_key: Optional[str] = None):
        # Load environment variables if not provided
        if not api_key:
            import os
            from dotenv import load_dotenv
            load_dotenv("config.env")
            api_key = os.getenv("NEWSAPI_KEY")
            
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"
        logger.info("NewsAPI fetcher initialized", has_api_key=bool(self.api_key))
        
    async def fetch_trending_topic(self, query: str = None, sources: str = None) -> List[NewsArticle]:
        """Fetch articles about a trending/controversial topic"""
        if not self.api_key:
            logger.error("No NewsAPI key provided - cannot fetch articles")
            return []
            
        # Use current hot topics if no query specified
        if not query:
            query = await self._get_current_controversial_topic()
            
        logger.info(f"Fetching REAL articles for query: '{query}' using NewsAPI")
        
        url = f"{self.base_url}/everything"
        params = {
            'q': query,
            'sortBy': 'publishedAt',  # Get latest articles
            'pageSize': 100,  # Get more to filter from
            'language': 'en',
            'from': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),  # Last week
            'apiKey': self.api_key
        }
        
        if sources:
            params['sources'] = sources
            
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    data = await response.json()
                    
                    if data.get('status') != 'ok':
                        logger.error("NewsAPI error", error=data.get('message'))
                        return []
                        
                    raw_articles = data.get('articles', [])
                    logger.info(f"NewsAPI returned {len(raw_articles)} raw articles")
                    
                    articles = []
                    for article_data in raw_articles:
                        # Get content from content field or description
                        content = article_data.get('content') or article_data.get('description', '')
                        title = article_data.get('title', '')
                        
                        # Filter out removed/invalid articles and ensure substantial content
                        if (content and title and 
                            len(content) > 50 and 
                            content != '[Removed]' and 
                            title != '[Removed]' and
                            not content.endswith('[+chars]')):  # NewsAPI truncation marker
                            
                            article = NewsArticle(
                                title=title,
                                content=content,
                                source=article_data['source']['name'],
                                url=article_data['url'],
                                published_at=article_data['publishedAt']
                            )
                            articles.append(article)
                            
                            # Stop when we have enough articles
                            if len(articles) >= 20:
                                break
                    
                    logger.info(f"Filtered to {len(articles)} articles with valid content for query: '{query}'")
                    return articles
                    
        except Exception as e:
            logger.error("Failed to fetch from NewsAPI", error=str(e))
            return []
    
    async def _get_current_controversial_topic(self) -> str:
        """Get a current controversial topic from trending searches"""
        # These are typically controversial topics that generate diverse coverage
        current_topics = [
            "trump election",
            "biden administration", 
            "ukraine war",
            "climate change",
            "immigration policy",
            "economy inflation",
            "supreme court",
            "artificial intelligence"
        ]
        
        # For demo, rotate through topics based on current hour
        topic_index = datetime.now().hour % len(current_topics)
        return current_topics[topic_index]
    
    # All demo/fallback articles removed - system now only uses real NewsAPI data


class RSSNewsFetcher:
    """Alternative fetcher using RSS feeds from major news sources"""
    
    def __init__(self):
        self.rss_feeds = {
            'BBC': 'http://feeds.bbci.co.uk/news/rss.xml',
            'CNN': 'http://rss.cnn.com/rss/edition.rss',
            'Fox News': 'http://feeds.foxnews.com/foxnews/latest',
            'Reuters': 'http://feeds.reuters.com/reuters/topNews',
            'NPR': 'https://feeds.npr.org/1001/rss.xml',
            'The Guardian': 'https://www.theguardian.com/world/rss',
            'Wall Street Journal': 'https://feeds.a.dj.com/rss/RSSWorldNews.xml'
        }
    
    async def fetch_diverse_coverage(self, max_articles: int = 15) -> List[NewsArticle]:
        """Fetch recent articles from diverse sources for comparison"""
        all_articles = []
        
        for source_name, feed_url in list(self.rss_feeds.items())[:5]:  # Limit to 5 sources for speed
            try:
                feed = await asyncio.to_thread(feedparser.parse, feed_url)
                
                for entry in feed.entries[:3]:  # Max 3 per source
                    article = NewsArticle(
                        title=entry.title,
                        content=entry.get('summary', entry.get('description', '')),
                        source=source_name,
                        url=entry.link,
                        published_at=entry.get('published', datetime.now().isoformat())
                    )
                    all_articles.append(article)
                    
            except Exception as e:
                logger.warning(f"Failed to fetch RSS from {source_name}", error=str(e))
                continue
        
        logger.info(f"Fetched {len(all_articles)} articles from RSS feeds")
        return all_articles[:max_articles]
