// dashboard/lib/api/auth.ts
import { apiClient, ApiResponse } from './client';

const AUTH_ENDPOINTS = {
  LOGIN: '/auth/login',
  REGISTER: '/auth/register',
  LOGOUT: '/auth/logout',
  ME: '/auth/me',
  ALPACA_CONNECT: '/alpaca/connect',
};

export interface UserProfile {
  id: number;
  username: string;
  email: string;
  created_at: string;
  alpaca: {
    connected: boolean;
    status: 'active' | 'inactive' | 'error';
    message: string;
    has_credentials: boolean;
    last_verified: string | null;
  };
}

export interface LoginCredentials {
  username: string;
  password: string;
}

export interface RegisterData {
  username: string;
  email: string;
  password: string;
}

export interface AlpacaCredentials {
  alpaca_key: string;
  alpaca_secret: string;
  paper: boolean;
}

// User authentication
export const login = (credentials: LoginCredentials): ApiResponse<{ success: boolean; user: unknown }> => {
  return apiClient.post(AUTH_ENDPOINTS.LOGIN, credentials).then(res => res.data);
};

export const register = (data: RegisterData): ApiResponse<{ success: boolean; user: unknown }> => {
  return apiClient.post(AUTH_ENDPOINTS.REGISTER, data).then(res => res.data);
};

export const logout = (): ApiResponse<{ success: boolean }> => {
  return apiClient.post(AUTH_ENDPOINTS.LOGOUT).then(res => res.data);
};

export const getCurrentUser = (): ApiResponse<UserProfile> => {
  return apiClient.get(AUTH_ENDPOINTS.ME).then(res => res.data);
};

// Alpaca API connection
export const connectAlpaca = (credentials: AlpacaCredentials): ApiResponse<{ success: boolean; message: string }> => {
  return apiClient.post(AUTH_ENDPOINTS.ALPACA_CONNECT, credentials).then(res => res.data);
};