import { invoke } from '@tauri-apps/api/tauri';
import { Body, fetch as tauriFetch } from '@tauri-apps/api/http';

// Store auth token in memory for the session
let authToken: string | null = null;

export function setAuthToken(token: string | null) {
  authToken = token;
}

export function getAuthToken(): string | null {
  return authToken;
}

// Auth API adapter
export const authApi = {
  async login(credentials: { username: string; password: string }) {
    const response = await invoke<{
      success: boolean;
      user: any;
      token: string | null;
      error: string | null;
    }>('login', { credentials });
    
    if (response.success && response.token) {
      setAuthToken(response.token);
    }
    
    return response;
  },
  
  async getCurrentUser() {
    if (!authToken) throw new Error('Not authenticated');
    
    return await invoke('get_current_user', { token: authToken });
  },
  
  async logout() {
    setAuthToken(null);
    return { success: true };
  },
  
  async connectAlpaca(credentials: any) {
    const user = await this.getCurrentUser();
    if (!user) throw new Error('Not authenticated');
    
    return await invoke('update_alpaca_credentials', {
      userId: user.id,
      credentials,
    });
  },
};

// Alpaca API adapter using Tauri's HTTP client
export const alpacaApi = {
  async request(endpoint: string, options: any = {}) {
    const user = await authApi.getCurrentUser();
    if (!user?.alpaca_key || !user?.alpaca_secret) {
      throw new Error('Alpaca credentials not configured');
    }
    
    // Use Tauri command to proxy Alpaca API calls
    const response = await invoke<string>('alpaca_api_call', {
      endpoint,
      method: options.method || 'GET',
      body: options.body ? JSON.stringify(options.body) : null,
      alpacaKey: user.alpaca_key,
      alpacaSecret: user.alpaca_secret,
    });
    
    return JSON.parse(response);
  },
  
  async getAccount() {
    return this.request('/v2/account');
  },
  
  async getPositions() {
    return this.request('/v2/positions');
  },
  
  async getOrders(params?: any) {
    const query = params ? `?${new URLSearchParams(params)}` : '';
    return this.request(`/v2/orders${query}`);
  },
  
  async createOrder(order: any) {
    return this.request('/v2/orders', {
      method: 'POST',
      body: order,
    });
  },
  
  async cancelOrder(orderId: string) {
    return this.request(`/v2/orders/${orderId}`, {
      method: 'DELETE',
    });
  },
  
  async getPortfolioHistory(period: string, timeframe: string) {
    return this.request(`/v2/account/portfolio/history?period=${period}&timeframe=${timeframe}`);
  },
};

// Portfolio service adapter (for the FastAPI backend)
export const portfolioApi = {
  async getRecommendations(params: any) {
    // This can still use regular HTTP since it's a local service
    const response = await tauriFetch('http://localhost:8001/api/portfolio/recommendations', {
      method: 'GET',
      query: params,
    });
    
    return response.data;
  },
};

