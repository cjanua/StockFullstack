// dashboard/lib/api/alpaca.ts
import { apiClient, ApiResponse } from './client';
import { Position, Account, PortfolioHistory, Watchlist } from '@/types/alpaca';

const ALPACA_ENDPOINTS = {
  ACCOUNT: '/alpaca/account',
  POSITIONS: '/alpaca/positions',
  ORDERS: '/alpaca/orders',
  PORTFOLIO_HISTORY: '/alpaca/account/history',
  WATCHLISTS: '/alpaca/account/watchlists',
  PORTFOLIO_RECOMMENDATIONS: '/alpaca/portfolio/recommendations',
};

export interface PortfolioRecommendationsParams {
  lookback_days?: number;
  min_change_percent?: number;
  cash_reserve_percent?: number;
}

export interface PortfolioRecommendation {
  symbol: string;
  current_shares: number;
  target_shares: number;
  difference: number;
  action: 'Buy' | 'Sell';
  quantity: number;
}

export interface PortfolioRecommendationsResponse {
  portfolio_value: number;
  cash: number;
  target_cash: number;
  recommendations: PortfolioRecommendation[];
}

// Account endpoints
export const getAccount = (): ApiResponse<Account> => {
  return apiClient.get(ALPACA_ENDPOINTS.ACCOUNT).then(res => res.data);
};

// Positions endpoints
export const getPositions = (): ApiResponse<Position[]> => {
  return apiClient.get(ALPACA_ENDPOINTS.POSITIONS).then(res => res.data);
};

export const closePosition = (symbol: string): ApiResponse<any> => {
  return apiClient.delete(`${ALPACA_ENDPOINTS.POSITIONS}/${symbol}`).then(res => res.data);
};

// History endpoints
export const getPortfolioHistory = (days: number, timeframe: string): ApiResponse<PortfolioHistory> => {
  return apiClient.get(ALPACA_ENDPOINTS.PORTFOLIO_HISTORY, {
    headers: {
      days: days.toString(),
      timeframe: timeframe,
    },
  }).then(res => res.data);
};

// Watchlists endpoints
export const getWatchlists = (): ApiResponse<Watchlist[]> => {
  return apiClient.get(ALPACA_ENDPOINTS.WATCHLISTS).then(res => res.data);
};

// Orders endpoints
export const getOrders = (params: any = {}): ApiResponse<any[]> => {
  return apiClient.get(ALPACA_ENDPOINTS.ORDERS, { params }).then(res => res.data);
};

export const createOrder = (orderData: any): ApiResponse<any> => {
  return apiClient.post(ALPACA_ENDPOINTS.ORDERS, orderData).then(res => res.data);
};

export const cancelOrder = (orderId: string): ApiResponse<any> => {
  return apiClient.delete(`${ALPACA_ENDPOINTS.ORDERS}?id=${orderId}`).then(res => res.data);
};

export const cancelAllOrders = (): ApiResponse<any> => {
  return apiClient.delete(ALPACA_ENDPOINTS.ORDERS).then(res => res.data);
};

// Portfolio optimization endpoints
export const getPortfolioRecommendations = (
  params: PortfolioRecommendationsParams = {}
): ApiResponse<PortfolioRecommendationsResponse> => {
  return apiClient.get(ALPACA_ENDPOINTS.PORTFOLIO_RECOMMENDATIONS, { params })
    .then(res => res.data);
};