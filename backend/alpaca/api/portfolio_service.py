# backend/alpaca/api/portfolio_service.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pypfopt import EfficientFrontier, expected_returns, risk_models
from typing import Dict
import asyncio
import logging
import time  # Add import for tracking recently closed positions
import numpy as np
from functools import lru_cache
from datetime import datetime

from backend.alpaca.core import AlpacaConfig
from backend.alpaca.sdk.loaders import (
    get_account, 
    get_positions, 
    get_history,
    get_trading_client,
    clear_portfolio_cache  # Import the new function
)
from result import Ok, Err

ALPACA_KEY, ALPACA_SECRET, _ = AlpacaConfig().get_credentials()

app = FastAPI(title="Portfolio Optimization Service")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add this to your FastAPI app setup
@app.middleware("http")
async def log_requests(request, call_next):
    """Log all incoming requests"""
    logger.info(f"Request: {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

# Add this before the existing health check endpoint
@app.get("/")
async def root():
    """Root endpoint that lists all available endpoints"""
    routes = []
    for route in app.routes:
        if hasattr(route, "path"):
            routes.append(f"{route.path}")
    
    return {
        "status": "Portfolio Optimization Service is running",
        "endpoints": routes
    }

class PortfolioRecommendation:
    def __init__(self, symbol: str, current_shares: float, target_shares: float):
        self.symbol = symbol
        self.current_shares = current_shares
        self.target_shares = target_shares
        self.difference = target_shares - current_shares
        self.action = "Buy" if self.difference > 0 else "Sell"
        self.quantity = abs(self.difference)

# Add performance metrics for monitoring
PERFORMANCE_METRICS = {
    'history_fetch_time': [],
    'optimization_time': [],
    'recommendation_time': []
}

@lru_cache(maxsize=4)  # Cache the 4 most recent optimization results in memory
def cached_get_optimal_portfolio(lookback_days: int):
    """Memory-cached version for frequent similar requests"""
    # This is just a wrapper to enable memoization
    # The actual implementation will call the real function
    # The timestamp prevents caching forever
    current_hour = datetime.now().strftime('%Y-%m-%d-%H')
    return lookback_days, current_hour

async def get_optimal_portfolio(lookback_days: int = 365):
    """Calculate optimal portfolio weights with performance tracking"""
    # Check in-memory cache first (the function returns a tuple but we only care about lookback_days)
    cached_key = cached_get_optimal_portfolio(lookback_days)
    
    try:
        # Measure performance
        start_time = time.time()
        
        history_result = await asyncio.to_thread(get_history, lookback_days)
        history_fetch_time = time.time() - start_time
        PERFORMANCE_METRICS['history_fetch_time'].append(history_fetch_time)
        logger.info(f"History fetch took {history_fetch_time:.2f} seconds for {lookback_days} days")
        
        if history_result.is_err():
            raise HTTPException(status_code=500, detail=f"Error fetching historical data: {history_result.err_value}")
        
        history = history_result.ok_value
        logger.info(f"Fetched historical data with shape: {history.shape} and columns: {list(history.columns)}")
        
        if history is None or history.empty:
            raise HTTPException(status_code=500, detail="No historical data available")
        
        # Start optimization timing
        opt_start_time = time.time()
        
        # Use the entire DataFrame as close prices
        close_prices = history
        
        # Fill missing values
        close_prices = close_prices.ffill().bfill()
        
        # Ensure we have enough data
        if len(close_prices) < 30:
            raise HTTPException(status_code=500, detail=f"Not enough historical data: only {len(close_prices)} data points")
        
        # Calculate expected returns and covariance matrix
        # For very large datasets, use more efficient calculation methods
        if lookback_days > 1825:  # More than 5 years
            # Use exponential weighted methods for large datasets
            mu = expected_returns.ema_historical_return(close_prices)
            S = risk_models.exp_cov(close_prices)
        else:
            mu = expected_returns.mean_historical_return(close_prices)
            S = risk_models.sample_cov(close_prices)
        
        # Optimize for maximum Sharpe ratio
        ef = EfficientFrontier(mu, S)
        weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        
        opt_time = time.time() - opt_start_time
        PERFORMANCE_METRICS['optimization_time'].append(opt_time)
        logger.info(f"Portfolio optimization took {opt_time:.2f} seconds for {lookback_days} days")
        
        # Log the optimized weights for debugging
        logger.info(f"Optimized weights: {cleaned_weights}")
        
        return cleaned_weights
    
    except Exception as e:
        logger.error(f"Error in get_optimal_portfolio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in get_optimal_portfolio: {str(e)}")

@app.get("/api/portfolio/optimize", response_model=Dict[str, float])
async def optimize_portfolio(lookback_days: int = Query(365, description="Days of historical data to use")):
    """Get optimized portfolio weights"""
    return await get_optimal_portfolio(lookback_days)

# Add a cache to track recently closed positions
# Format: {"SYMBOL": timestamp_closed}
RECENTLY_CLOSED_POSITIONS = {}
COOLING_PERIOD = 300  # 5 minutes in seconds

# Add endpoint to record position closures
@app.post("/api/portfolio/record-position-close/{symbol}")
async def record_position_close(symbol: str):
    """Record that a position was recently closed to prevent immediate re-recommendation"""
    symbol = symbol.upper()
    RECENTLY_CLOSED_POSITIONS[symbol] = time.time()
    logger.info(f"Recorded closure of position {symbol}")
    return {"status": "success", "message": f"Recorded closure of {symbol}"}

@app.get("/api/portfolio/recommendations")
async def get_recommendations(
    lookback_days: int = Query(365, description="Days of historical data to use"),
    min_change_percent: float = Query(0.01, description="Minimum position change percentage to recommend"),
    cash_reserve_percent: float = Query(0.05, description="Cash percentage to keep in reserve (0-1)")
):
    """Get buy/sell recommendations to optimize your portfolio with performance tracking"""
    start_time = time.time()
    
    # Clean up expired entries in recently closed positions
    current_time = time.time()
    expired_symbols = [symbol for symbol, timestamp in RECENTLY_CLOSED_POSITIONS.items() 
                      if current_time - timestamp > COOLING_PERIOD]
    for symbol in expired_symbols:
        del RECENTLY_CLOSED_POSITIONS[symbol]
    
    # Get current account info and positions
    account_result = get_account()
    if account_result.is_err():
        raise HTTPException(status_code=500, detail=f"Error fetching account: {account_result.err_value}")
    account = account_result.ok_value
    
    positions_result = get_positions()
    if positions_result.is_err():
        raise HTTPException(status_code=500, detail=f"Error fetching positions: {positions_result.err_value}")
    positions = positions_result.ok_value
    
    # Calculate optimal weights
    optimal_weights = await get_optimal_portfolio(lookback_days)
    
    # Remove recently closed positions from optimal weights
    for symbol in list(RECENTLY_CLOSED_POSITIONS.keys()):
        if symbol in optimal_weights:
            logger.info(f"Removing recently closed position {symbol} from recommendations")
            del optimal_weights[symbol]
    
    # Get account values
    portfolio_value = float(account['portfolio_value'])
    cash = float(account['cash'])
    
    # Process current positions
    current_positions = {}
    for pos in positions:
        symbol = pos['symbol']
        qty = float(pos['qty'])
        current_price = float(pos['current_price'])
        market_value = float(pos['market_value'])
        
        current_positions[symbol] = {
            'shares': qty,
            'price': current_price,
            'value': market_value,
            'weight': market_value / portfolio_value
        }
    
    # Adjust for cash reserve
    investable_amount = portfolio_value * (1 - cash_reserve_percent)
    
    # Calculate target position values and shares
    recommendations = []
    
    # First, handle existing positions
    for symbol, position in current_positions.items():
        current_shares = position['shares']
        current_price = position['price']
        
        # If symbol is in optimal weights
        if symbol in optimal_weights:
            target_value = investable_amount * optimal_weights[symbol]
            target_shares = target_value / current_price
            
            # Only recommend changes above threshold
            pct_change = abs(target_shares - current_shares) / current_shares if current_shares > 0 else 1
            if pct_change > min_change_percent:
                recommendations.append(PortfolioRecommendation(
                    symbol=symbol,
                    current_shares=current_shares,
                    target_shares=target_shares
                ))
        else:
            # Position should be sold
            recommendations.append(PortfolioRecommendation(
                symbol=symbol,
                current_shares=current_shares,
                target_shares=0
            ))
    
    # Handle new positions (in optimal weights but not current portfolio)
    for symbol, weight in optimal_weights.items():
        if symbol not in current_positions and weight > min_change_percent:
            # Get current price for the symbol
            # This would require a separate API call in a production environment
            # For simplicity, we'll use a placeholder
            try:
                quote_result = await asyncio.to_thread(get_quote_for_symbol, symbol)
                if quote_result.is_err():
                    continue  # Skip if we can't get a quote
                
                current_price = float(quote_result.ok_value['current_price'])
                target_value = investable_amount * weight
                target_shares = target_value / current_price
                
                recommendations.append(PortfolioRecommendation(
                    symbol=symbol,
                    current_shares=0,
                    target_shares=target_shares
                ))
            except Exception:
                # Skip symbols we can't price
                continue
    
    # Sort recommendations: buys first, then sells
    recommendations.sort(key=lambda x: (x.action != "Buy", abs(x.difference)))
    
    # Track total recommendation time
    total_time = time.time() - start_time
    PERFORMANCE_METRICS['recommendation_time'].append(total_time)
    logger.info(f"Complete recommendation process took {total_time:.2f} seconds for {lookback_days} days")
    
    return {
        "portfolio_value": portfolio_value,
        "cash": cash,
        "target_cash": portfolio_value * cash_reserve_percent,
        "recommendations": [vars(r) for r in recommendations],
        "processing_time_seconds": round(total_time, 2)  # Add processing time to response
    }

async def get_quote_for_symbol(symbol):
    """Get the current price for a symbol"""
    client_res = get_trading_client()
    if not client_res.is_ok():
        return Err(client_res.err_value)
    client = client_res.ok_value
    
    try:
        # Get last quote
        quote = client.get_latest_trade(symbol)
        return Ok({
            'symbol': symbol,
            'current_price': str(quote.price)
        })
    except Exception as e:
        return Err(f"Failed to get quote for {symbol}: {str(e)}")

# Add a simple health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/api/debug/history")
async def debug_history(lookback_days: int = Query(30, description="Days of historical data to use")):
    """Debug endpoint to see what historical data looks like"""
    history_result = await asyncio.to_thread(get_history, lookback_days)
    if history_result.is_err():
        return {"error": f"Error fetching historical data: {history_result.err_value}"}
    
    history = history_result.ok_value
    if history is None or history.empty:
        return {"error": "No historical data available"}
    
    # Return information about the dataframe
    info = {
        "shape": history.shape,
        "columns": list(history.columns),
        "index": list(history.index[:5]),  # First 5 indices
        "sample": history.head(3).to_dict(),
        "has_multiindex": isinstance(history.columns, pd.MultiIndex),
    }
    
    if isinstance(history.columns, pd.MultiIndex):
        info["multiindex_levels"] = [list(level) for level in history.columns.levels]
    
    return info

@app.get("/api/debug/client")
async def debug_client():
    """Debug endpoint to check what client methods are available"""
    # Check trading client
    client_res = get_trading_client()
    if client_res.is_err():
        return {"error": client_res.err_value}
    
    client = client_res.ok_value
    
    # Get available methods and attributes
    trading_methods = [method for method in dir(client) if not method.startswith('_')]
    
    # Check for historical data client
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        # Import here to avoid circular imports
        from alpaca.data.timeframe import TimeFrame
        from alpaca.data.requests import StockBarsRequest
        
        # Get API keys from environment
        import os
        api_key = ALPACA_KEY
        api_secret = ALPACA_SECRET
        
        # Create a dedicated data client
        data_client = StockHistoricalDataClient(api_key, api_secret)
        data_methods = [method for method in dir(data_client) if not method.startswith('_')]
        
        # Check if the data client has the methods we need
        has_get_stock_bars = hasattr(data_client, 'get_stock_bars')
        
        return {
            "trading_client_type": type(client).__name__,
            "trading_client_methods": trading_methods[:20],  # Limit to first 20 for readability
            "data_client_type": type(data_client).__name__,
            "data_client_methods": data_methods,
            "has_get_stock_bars": has_get_stock_bars,
            "api_key_present": bool(api_key),
            "api_secret_present": bool(api_secret)
        }
    except Exception as e:
        return {
            "trading_client_type": type(client).__name__,
            "trading_client_methods": trading_methods[:20],
            "data_client_error": str(e),
            "api_key_present": bool(ALPACA_KEY),
            "api_secret_present": bool(ALPACA_SECRET)
        }

# Add a cache clear endpoint
@app.post("/api/portfolio/clear-cache")
async def clear_cache():
    """Force clear Redis cache for portfolio data"""
    try:
        # Call the function to clear cache
        await asyncio.to_thread(clear_portfolio_cache)
        return {"status": "success", "message": "Portfolio cache cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")

# Add an endpoint to clear historical data cache
@app.post("/api/portfolio/clear-history-cache")
async def clear_history_cache_endpoint(days: int = None):
    """Clear cached historical data to force refresh"""
    from backend.alpaca.sdk.loaders import clear_history_cache
    
    try:
        await asyncio.to_thread(clear_history_cache, days)
        return {"status": "success", "message": f"Historical data cache {'for ' + str(days) + ' days ' if days else ''}cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear history cache: {str(e)}")

# Add a performance monitoring endpoint
@app.get("/api/debug/performance")
async def get_performance_metrics():
    """Get performance metrics for optimization processes"""
    metrics = {}
    
    for key, values in PERFORMANCE_METRICS.items():
        if values:
            metrics[key] = {
                "count": len(values),
                "average": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "recent": values[-10:]  # Last 10 measurements
            }
        else:
            metrics[key] = {"count": 0}
    
    return metrics

# Add this after all your endpoint registrations
print("\n=== REGISTERED ROUTES ===")
for route in app.routes:
    print(f"  {route.path} [{','.join(route.methods)}]")
print("========================\n")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)