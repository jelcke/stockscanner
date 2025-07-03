"""
News and Social Media Sentiment Analysis Integration
Combines sentiment data with stock scanning
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List

import aiohttp
from textblob import TextBlob

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self, news_api_key: str = None):
        self.news_api_key = news_api_key
        self.sentiment_cache = {}

    def analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text using TextBlob"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1

            # Convert to more interpretable scores
            sentiment_score = (polarity + 1) / 2  # 0 to 1
            confidence = 1 - subjectivity  # Higher confidence for objective text

            return {
                'sentiment_score': sentiment_score,
                'confidence': confidence,
                'polarity': polarity,
                'subjectivity': subjectivity
            }

        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {'sentiment_score': 0.5, 'confidence': 0.0, 'polarity': 0.0, 'subjectivity': 1.0}

    async def get_news_sentiment(self, symbol: str, hours_back: int = 24) -> Dict[str, float]:
        """Get news sentiment for a symbol"""
        if not self.news_api_key:
            logger.warning("No news API key provided")
            return {'sentiment_score': 0.5, 'confidence': 0.0, 'article_count': 0}

        cache_key = f"{symbol}_{hours_back}"
        if cache_key in self.sentiment_cache:
            cache_time, data = self.sentiment_cache[cache_key]
            if (datetime.now() - cache_time).seconds < 1800:  # 30 min cache
                return data

        try:
            from_date = (datetime.now() - timedelta(hours=hours_back)).strftime('%Y-%m-%d')

            url = "https://newsapi.org/v2/everything"
            params = {
                'q': f'"{symbol}" OR "{self._get_company_name(symbol)}"',
                'from': from_date,
                'sortBy': 'relevancy',
                'apiKey': self.news_api_key,
                'language': 'en',
                'pageSize': 50
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    data = await response.json()

            articles = data.get('articles', [])
            if not articles:
                return {'sentiment_score': 0.5, 'confidence': 0.0, 'article_count': 0}

            sentiments = []
            confidences = []

            for article in articles:
                title = article.get('title', '')
                description = article.get('description', '')
                text = f"{title} {description}"

                if text.strip():
                    sentiment_data = self.analyze_text_sentiment(text)
                    sentiments.append(sentiment_data['sentiment_score'])
                    confidences.append(sentiment_data['confidence'])

            if sentiments:
                # Weighted average by confidence
                weights = [c if c > 0 else 0.1 for c in confidences]
                avg_sentiment = sum(s * w for s, w in zip(sentiments, weights)) / sum(weights)
                avg_confidence = sum(confidences) / len(confidences)
            else:
                avg_sentiment = 0.5
                avg_confidence = 0.0

            result = {
                'sentiment_score': avg_sentiment,
                'confidence': avg_confidence,
                'article_count': len(articles)
            }

            # Cache result
            self.sentiment_cache[cache_key] = (datetime.now(), result)

            return result

        except Exception as e:
            logger.error(f"News sentiment fetch failed for {symbol}: {e}")
            return {'sentiment_score': 0.5, 'confidence': 0.0, 'article_count': 0}

    def _get_company_name(self, symbol: str) -> str:
        """Get company name from symbol (simplified mapping)"""
        symbol_map = {
            'AAPL': 'Apple',
            'MSFT': 'Microsoft',
            'GOOGL': 'Google',
            'AMZN': 'Amazon',
            'TSLA': 'Tesla',
            'META': 'Meta',
            'NVDA': 'NVIDIA'
        }
        return symbol_map.get(symbol, symbol)

    def calculate_sentiment_momentum(self, symbol: str) -> float:
        """Calculate sentiment momentum (recent vs older sentiment)"""
        try:
            recent_sentiment = asyncio.run(self.get_news_sentiment(symbol, hours_back=6))
            older_sentiment = asyncio.run(self.get_news_sentiment(symbol, hours_back=24))

            recent_score = recent_sentiment['sentiment_score']
            older_score = older_sentiment['sentiment_score']

            # Calculate momentum (-1 to 1)
            momentum = (recent_score - older_score) * 2
            return max(-1, min(1, momentum))

        except Exception as e:
            logger.error(f"Sentiment momentum calculation failed for {symbol}: {e}")
            return 0.0

class SentimentScanner:
    def __init__(self, sentiment_analyzer: SentimentAnalyzer):
        self.analyzer = sentiment_analyzer
        self.sentiment_data = {}

    async def update_sentiment_data(self, symbols: List[str]):
        """Update sentiment data for all symbols"""
        tasks = []
        for symbol in symbols:
            task = self._update_symbol_sentiment(symbol)
            tasks.append(task)

        await asyncio.gather(*tasks, return_exceptions=True)

    async def _update_symbol_sentiment(self, symbol: str):
        """Update sentiment data for a single symbol"""
        try:
            sentiment_data = await self.analyzer.get_news_sentiment(symbol)
            momentum = self.analyzer.calculate_sentiment_momentum(symbol)

            self.sentiment_data[symbol] = {
                'sentiment_score': sentiment_data['sentiment_score'],
                'confidence': sentiment_data['confidence'],
                'article_count': sentiment_data['article_count'],
                'momentum': momentum,
                'last_update': datetime.now()
            }

        except Exception as e:
            logger.error(f"Sentiment update failed for {symbol}: {e}")

    def get_sentiment_filter(self, min_sentiment=0.6, min_confidence=0.3, min_articles=3) -> List[str]:
        """Get symbols with positive sentiment"""
        filtered_symbols = []

        for symbol, data in self.sentiment_data.items():
            if (data['sentiment_score'] >= min_sentiment and
                data['confidence'] >= min_confidence and
                data['article_count'] >= min_articles):
                filtered_symbols.append(symbol)

        return filtered_symbols

    def get_sentiment_alerts(self, momentum_threshold=0.3) -> List[Dict]:
        """Get sentiment-based alerts"""
        alerts = []

        for symbol, data in self.sentiment_data.items():
            if abs(data['momentum']) >= momentum_threshold:
                alert_type = "POSITIVE" if data['momentum'] > 0 else "NEGATIVE"

                alerts.append({
                    'symbol': symbol,
                    'type': f"{alert_type}_SENTIMENT",
                    'sentiment_score': data['sentiment_score'],
                    'momentum': data['momentum'],
                    'confidence': data['confidence'],
                    'article_count': data['article_count'],
                    'message': f"{symbol} showing {alert_type.lower()} sentiment momentum"
                })

        return sorted(alerts, key=lambda x: abs(x['momentum']), reverse=True)

    def print_sentiment_summary(self):
        """Print sentiment analysis summary"""
        print(f"\n{'='*70}")
        print(f"Sentiment Analysis Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")
        print(f"{'Symbol':<8} {'Sentiment':<12} {'Momentum':<12} {'Articles':<10} {'Confidence':<12}")
        print(f"{'-'*70}")

        for symbol, data in sorted(self.sentiment_data.items(),
                                 key=lambda x: x[1]['sentiment_score'], reverse=True):
            sentiment_str = f"{data['sentiment_score']:.2f}"
            momentum_str = f"{data['momentum']:+.2f}"
            articles_str = str(data['article_count'])
            confidence_str = f"{data['confidence']:.2f}"

            print(f"{symbol:<8} {sentiment_str:<12} {momentum_str:<12} "
                  f"{articles_str:<10} {confidence_str:<12}")

# Example usage
async def demo_sentiment_analysis():
    """Demonstrate sentiment analysis without API key"""
    print("Sentiment Analysis Demo")
    print("="*50)
    print("\nNote: This is a demo without real news data.")
    print("To use real data, add a News API key.\n")

    # Create analyzer without API key (will use demo data)
    analyzer = SentimentAnalyzer()
    scanner = SentimentScanner(analyzer)

    # Demo text analysis
    sample_texts = {
        'AAPL': "Apple reports record-breaking iPhone sales and strong earnings beat. The company shows incredible growth momentum.",
        'TSLA': "Tesla faces production challenges and regulatory concerns. Analysts worry about competition.",
        'NVDA': "NVIDIA announces breakthrough AI chip technology. Stock soars on strong demand.",
        'META': "Meta's metaverse investments show mixed results. Investors remain cautiously optimistic."
    }

    print("Sample Text Analysis:")
    print("-"*50)

    for symbol, text in sample_texts.items():
        sentiment = analyzer.analyze_text_sentiment(text)
        print(f"\n{symbol}: {text[:60]}...")
        print(f"  Sentiment Score: {sentiment['sentiment_score']:.2f} (0=negative, 1=positive)")
        print(f"  Polarity: {sentiment['polarity']:+.2f}")
        print(f"  Confidence: {sentiment['confidence']:.2f}")

    # Simulate sentiment data
    print("\n\nSimulated Market Sentiment:")
    print("-"*50)

    scanner.sentiment_data = {
        'AAPL': {
            'sentiment_score': 0.75,
            'confidence': 0.8,
            'article_count': 15,
            'momentum': 0.2,
            'last_update': datetime.now()
        },
        'TSLA': {
            'sentiment_score': 0.35,
            'confidence': 0.6,
            'article_count': 25,
            'momentum': -0.3,
            'last_update': datetime.now()
        },
        'NVDA': {
            'sentiment_score': 0.85,
            'confidence': 0.9,
            'article_count': 10,
            'momentum': 0.5,
            'last_update': datetime.now()
        },
        'META': {
            'sentiment_score': 0.55,
            'confidence': 0.5,
            'article_count': 8,
            'momentum': 0.1,
            'last_update': datetime.now()
        }
    }

    scanner.print_sentiment_summary()

    # Show alerts
    alerts = scanner.get_sentiment_alerts(momentum_threshold=0.2)
    if alerts:
        print("\n\nSentiment Alerts:")
        print("-"*50)
        for alert in alerts:
            print(f"\n{alert['symbol']}: {alert['message']}")
            print(f"  Sentiment: {alert['sentiment_score']:.2f}")
            print(f"  Momentum: {alert['momentum']:+.2f}")
            print(f"  Articles: {alert['article_count']}")

if __name__ == "__main__":
    # Install textblob if needed
    try:
        import textblob
    except ImportError:
        print("Installing textblob...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "textblob"])

    asyncio.run(demo_sentiment_analysis())
