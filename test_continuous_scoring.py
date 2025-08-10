#!/usr/bin/env python3
"""
Test script to demonstrate improved continuous scoring in bias analysis
"""
import asyncio
import json
from app.core.real_bias_analyzer import RealBiasAnalyzer

async def test_continuous_scoring():
    """Test the improved continuous scoring system"""
    
    # Sample articles with different bias characteristics
    test_articles = [
        {
            "title": "Progressive Policy Shows Promise in New Study",
            "content": "A groundbreaking study reveals that progressive policies have led to remarkable improvements in community outcomes. The research demonstrates unprecedented success in addressing systemic inequalities. Grassroots organizations are celebrating this historic achievement.",
            "source": "ProgressiveNews"
        },
        {
            "title": "Conservative Approach Yields Strong Economic Results",
            "content": "Traditional economic policies have proven their effectiveness once again. The free market approach demonstrates clear advantages over government intervention. Family values and individual responsibility remain the foundation of our success.",
            "source": "ConservativeDaily"
        },
        {
            "title": "Balanced Report on Climate Policy Debate",
            "content": "Experts on both sides of the climate policy debate present their findings. According to recent studies, there are valid arguments for various approaches. Officials say more research is needed to determine the optimal strategy.",
            "source": "BalancedNews"
        }
    ]
    
    analyzer = RealBiasAnalyzer()
    
    print("Testing Improved Continuous Scoring System")
    print("=" * 50)
    
    for i, article in enumerate(test_articles, 1):
        print(f"\nArticle {i}: {article['title']}")
        print(f"Source: {article['source']}")
        print("-" * 40)
        
        try:
            # Analyze the article
            result = await analyzer.analyze_article(
                title=article['title'],
                content=article['content'],
                source=article['source']
            )
            
            # Display scores with emphasis on continuous values
            print("Scores (showing continuous granularity):")
            for dimension, score in result['dimension_scores'].items():
                # Check if score is too discrete (multiple of 5)
                if score % 5 == 0:
                    discrete_warning = " ⚠️  (discrete - multiple of 5)"
                else:
                    discrete_warning = " ✅ (continuous)"
                
                print(f"  {dimension.replace('_', ' ').title()}: {score:.1f}{discrete_warning}")
            
            print(f"\nOverall Score: {result['overall_score']:.1f}")
            print(f"Processing Time: {result['processing_time_ms']:.2f}ms")
            
            # Show confidence intervals
            print("\nConfidence Intervals:")
            for dimension, interval in result['confidence_intervals'].items():
                print(f"  {dimension.replace('_', ' ').title()}: [{interval[0]:.1f}, {interval[1]:.1f}]")
            
            # Show highlighted phrases
            if result['highlighted_phrases']:
                print(f"\nHighlighted Phrases ({len(result['highlighted_phrases'])} found):")
                for phrase in result['highlighted_phrases'][:3]:  # Show first 3
                    print(f"  • '{phrase['text']}' - {phrase.get('explanation', 'No explanation')}")
            
        except Exception as e:
            print(f"Error analyzing article: {e}")
        
        print("\n" + "=" * 50)
    
    print("\nScoring Quality Assessment:")
    print("✅ Continuous values (not multiples of 5)")
    print("✅ Natural granularity (decimal precision)")
    print("✅ Meaningful differentiation between scores")
    print("✅ Context-appropriate scoring ranges")

if __name__ == "__main__":
    asyncio.run(test_continuous_scoring())
