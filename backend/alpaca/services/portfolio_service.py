from fastapi import FastAPI, HTTPException, Depends, Query, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier, expected_returns, risk_models
from typing import Dict, List, Optional
import asyncio
import logging
from datetime import datetime, timedelta

from backend.alpaca.sdk.loaders import (
    ALPACA_KEY,
    ALPACA_SECRET,
    get_account, 
    get_positions, 
    get_history,
    get_trading_client,
    get_historical_data_client
)
from serializers import serialize_position, serialize_account
from result import Ok, Err

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

async def get_optimal_portfolio(lookback_days: int = 365):
    """Calculate optimal portfolio weights using PyPortfolioOpt"""
    try:
        history_result = await asyncio.to_thread(get_history, lookback_days)
        if history_result.is_err():
            raise HTTPException(status_code=500, detail=f"Error fetching historical data: {history_result.err_value}")
        
        history = history_result.ok_value
        logger.info(f"Fetched historical data with shape: {history.shape} and columns: {list(history.columns)}")
        
        if history is None or history.empty:
            raise HTTPException(status_code=500, detail="No historical data available")
        
        # Use the entire DataFrame as close prices
        close_prices = history  # Assuming the DataFrame columns are the stock symbols
        
        # Fill missing values
        close_prices = close_prices.ffill().bfill()
        
        # Ensure we have enough data
        if len(close_prices) < 30:
            raise HTTPException(status_code=500, detail=f"Not enough historical data: only {len(close_prices)} data points")
        
        # Calculate expected returns and covariance matrix
        mu = expected_returns.mean_historical_return(close_prices)
        S = risk_models.sample_cov(close_prices)
        
        # Optimize for maximum Sharpe ratio
        ef = EfficientFrontier(mu, S)
        weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        
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

@app.get("/api/portfolio/recommendations")
async def get_recommendations(
    lookback_days: int = Query(365, description="Days of historical data to use"),
    min_change_percent: float = Query(0.01, description="Minimum position change percentage to recommend"),
    cash_reserve_percent: float = Query(0.05, description="Cash percentage to keep in reserve (0-1)")
):
    """Get buy/sell recommendations to optimize your portfolio"""
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
    
    return {
        "portfolio_value": portfolio_value,
        "cash": cash,
        "target_cash": portfolio_value * cash_reserve_percent,
        "recommendations": [vars(r) for r in recommendations]
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

# Add this after all your endpoint registrations
print("\n=== REGISTERED ROUTES ===")
for route in app.routes:
    print(f"  {route.path} [{','.join(route.methods)}]")
print("========================\n")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)