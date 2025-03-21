import { useQuery } from '@tanstack/react-query';
import axios from 'axios';

interface MarketClockResponse {
  is_open: boolean;
  next_open?: string;
  next_close?: string;
  timestamp?: string;
}

export function useMarketHours() {
  const { data, isLoading, error } = useQuery<MarketClockResponse>({
    queryKey: ['marketClock'],
    queryFn: async () => {
      try {
        const response = await axios.get('/api/alpaca/market/clock');
        return response.data;
      } catch (error) {
        // Return default values on error to prevent UI from breaking
        return { is_open: true }; // Default to open on error
      }
    },
    refetchOnWindowFocus: false,
    staleTime: 5 * 60 * 1000,
    retry: 1,
    // Default to market open if query fails
    placeholderData: { is_open: true }
  });

  return {
    isMarketOpen: data?.is_open ?? true, // Default to open on undefined
    nextOpenTime: data?.next_open,
    nextCloseTime: data?.next_close,
    currentTime: data?.timestamp,
    isLoading,
    error,
  };
}
