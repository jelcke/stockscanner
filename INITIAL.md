## FEATURE:
Interactive Brokers Real-Time Stock Scanner with AI-Enhanced Pattern Recognition

Build a production-ready Python stock scanner that connects to Interactive Brokers API to identify fast-moving stocks with high percentage gains, unusual volume activity, and momentum breakouts. The system will include AI-powered pattern recognition to detect chart patterns, sentiment analysis integration, and machine learning models for predicting breakout probability. Based on the `ib_async` library, this scanner will be designed for high performance, real-time data processing, and advanced analytics.
Use live data from Interactve brokers API to scan for stocks that meet specific criteria such as:

## EXAMPLES:

### `examples/basic_scanner.py`
Basic implementation showing core IB API connectivity, market data subscription, and simple percentage gain filtering. Demonstrates the fundamental structure without AI components.

### `examples/pattern_detection.py` 
Example of AI pattern recognition using computer vision techniques to identify chart patterns like cup-and-handle, ascending triangles, and bull flags from candlestick data. Shows how to convert price data to image format and use CNN models for pattern classification.

### `examples/sentiment_integration.py`
Integration with news APIs and social media sentiment analysis. Demonstrates how to fetch real-time news for symbols, perform sentiment scoring, and incorporate sentiment data into scanning criteria.

### `examples/ml_breakout_predictor.py`
Machine learning model training and inference example using features like relative volume, price momentum, volatility, and technical indicators to predict breakout probability. Includes model training pipeline and real-time prediction integration.

### `examples/advanced_display.py`
Rich terminal interface with live updating tables, charts, and alerts. Shows how to create professional-looking output with color coding, progress bars, and real-time data visualization.

### `examples/backtesting_framework.py`
Historical backtesting system to validate scanner performance. Demonstrates how to test scanning criteria against historical data to measure accuracy and profitability.

## DOCUMENTATION:

### Interactive Brokers API Documentation
- **TWS API Reference**: https://interactivebrokers.github.io/tws-api/
- **Market Data Subscriptions**: https://www.interactivebrokers.com/campus/ibkr-api-page/market-data-subscriptions/
- **Python API Installation**: https://www.interactivebrokers.com/campus/ibkr-api-page/twsapi-doc/
- **ib_async Documentation**: https://ib-async.readthedocs.io/api.html

### Machine Learning and Pattern Recognition
- **Technical Analysis Patterns**: https://www.investopedia.com/articles/technical/112601.asp
- **Scikit-learn Documentation**: https://scikit-learn.org/stable/documentation.html
- **OpenCV for Pattern Recognition**: https://docs.opencv.org/4.x/
- **TensorFlow/Keras for Deep Learning**: https://www.tensorflow.org/guide

### Financial Data and APIs
- **Alpha Vantage API**: https://www.alphavantage.co/documentation/
- **News API**: https://newsapi.org/docs
- **Yahoo Finance API**: https://github.com/ranaroussi/yfinance
- **FRED Economic Data**: https://fred.stlouisfed.org/docs/api/

### Trading and Market Analysis
- **Volume Analysis**: https://www.investopedia.com/articles/technical/02/010702.asp
- **Gap Trading Strategies**: https://www.investopedia.com/articles/trading/05/playinggaps.asp
- **Relative Volume (RVOL)**: https://www.ig.com/en/trading-strategies/what-is-the-relative-volume-indicator-and-how-do-you-use-it-when-230904

### Python Libraries Documentation
- **Rich (Terminal Formatting)**: https://rich.readthedocs.io/en/stable/
- **Pandas (Data Analysis)**: https://pandas.pydata.org/docs/
- **NumPy (Numerical Computing)**: https://numpy.org/doc/
- **Redis (Caching)**: https://redis-py.readthedocs.io/en/stable/

## OTHER CONSIDERATIONS:

### IB API Gotchas and Common Issues
- **Rate Limiting**: IB API has a 50 messages/second limit. AI assistants often miss implementing proper throttling and queuing mechanisms for large watchlists.
- **Market Data Lines**: Each symbol consumes a market data line. Standard accounts get 100 lines, but AI assistants rarely account for this limitation when scaling to hundreds of symbols.
- **Connection Management**: IB Gateway requires persistent connections. Implement proper reconnection logic with exponential backoff - AI assistants often create naive retry loops.
- **Symbol Qualification**: Not all symbols are valid contracts. Always use `qualifyContractsAsync()` before subscribing to data.
- **Time Zones**: Market data timestamps are in exchange time zones. Implement proper timezone handling for multi-market scanning.

### Real-Time Data Challenges
- **Memory Management**: Continuous data streams can cause memory leaks. Implement circular buffers and periodic cleanup routines.
- **Threading Issues**: IB API uses callbacks that run in separate threads. AI assistants often miss thread-safety considerations when updating shared data structures.
- **Data Quality**: Handle invalid/NaN values in real-time feeds. Market data can contain outliers that break calculations.

### Machine Learning Considerations
- **Feature Engineering**: Raw price/volume data needs significant preprocessing. Include technical indicators, rolling statistics, and normalized features.
- **Look-Ahead Bias**: When training models, ensure no future data leaks into historical predictions. AI assistants commonly make this mistake in backtesting.
- **Model Drift**: Financial markets change - implement model retraining pipelines and performance monitoring.
- **Overfitting**: With thousands of potential features, models easily overfit. Use proper cross-validation and regularization.

### Production Deployment Issues
- **Error Recovery**: Financial systems need 99.9%+ uptime. Implement comprehensive error handling, logging, and automatic recovery mechanisms.
- **Security**: API keys, account credentials must be properly secured. Never hardcode credentials or log sensitive data.
- **Monitoring**: Include system health metrics, trade execution monitoring, and performance dashboards.
- **Compliance**: Ensure any automated trading complies with regulations. Log all decisions for audit trails.

### Performance Optimization
- **Database Design**: Use appropriate indexes for time-series data. Consider TimescaleDB for high-frequency data storage.
- **Caching Strategy**: Implement multi-layer caching (Redis for real-time, database for historical). Cache expensive calculations like technical indicators.
- **Async Programming**: Use asyncio properly for concurrent operations. AI assistants often mix blocking and non-blocking operations incorrectly.

### Testing and Validation
- **Paper Trading**: Always test with paper accounts first. Include paper trading mode in production code.
- **Unit Testing**: Mock IB API responses for reliable testing. Create fixtures for various market conditions.
- **Integration Testing**: Test with real IB Gateway in demo mode. Validate data accuracy against known sources.
- **Stress Testing**: Test behavior during market volatility, connection failures, and high-volume periods.

### AI-Specific Gotchas
- **Pattern Recognition**: Chart patterns are subjective. Train models on diverse, labeled datasets and validate with domain experts.
- **Sentiment Analysis**: Financial news sentiment is nuanced. Generic NLP models often fail on financial text - use domain-specific models.
- **Feature Scaling**: Different data types (prices, volumes, indicators) require different normalization approaches.
- **Real-Time Inference**: AI models must run in <100ms for real-time applications. Optimize model architecture and use efficient serving frameworks.