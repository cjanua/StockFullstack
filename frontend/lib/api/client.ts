/* eslint-disable @typescript-eslint/no-explicit-any */
// frontend/lib/api/client.ts
import axios, { AxiosInstance, AxiosRequestConfig, AxiosError, AxiosResponse } from 'axios';

// API error type for consistent error handling
export interface ApiError {
  message: string;
  code?: string;
  status?: number;
  details?: any;
}

// Create a custom API client with consistent error handling
export const createApiClient = (baseConfig: AxiosRequestConfig = {}): AxiosInstance => {
  const client = axios.create({
    baseURL: '/api',
    headers: {
      'Content-Type': 'application/json',
    },
    ...baseConfig,
  });

  // Add response interceptor for consistent error handling
  client.interceptors.response.use(
    (response: AxiosResponse) => response,
    (error: AxiosError) => {
      let apiError: ApiError = {
        message: 'An unexpected error occurred',
      };

      if (error.response) {
        // The request was made and the server responded with a status code
        // that falls out of the range of 2xx
        const data = error.response.data as any;
        apiError = {
          message: data.error || data.message || 'Server returned an error',
          status: error.response.status,
          details: data,
        };
      } else if (error.request) {
        // The request was made but no response was received
        apiError = {
          message: 'No response from server',
          details: error.request,
        };
      } else {
        // Something happened in setting up the request that triggered an Error
        apiError = {
          message: error.message || 'Failed to make request',
        };
      }

      return Promise.reject(apiError);
    }
  );

  return client;
};

// Create default API client
export const apiClient = createApiClient();

// Helper type for API endpoints to ensure consistent return types
export type ApiResponse<T> = Promise<T>;