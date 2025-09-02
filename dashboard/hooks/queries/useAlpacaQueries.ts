// dashboard/hooks/queries/useAlpacaQueries.ts
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { alpaca } from '@/lib/api';
import { auth } from '@/lib/api';
import { toast } from '@/hooks/use-toast';
import axios from 'axios';

// Query key factory for better organization and type safety
export const queryKeys = {
  account: ['account'] as const,
  positions: ['positions'] as const,
  portfolioHistory: (days: number, timeframe: string) => 
    ['portfolioHistory', days.toString(), timeframe] as const,
  watchlists: ['watchlists'] as const,
  orders: (status?: string) => ['orders', status ?? 'all'] as const,
  recommendations: (params: alpaca.PortfolioRecommendationsParams) => 
    ['portfolioRecommendations', JSON.stringify(params)] as const,
  user: ['user'] as const,
  quotes: (symbols: string[]) => ['quotes', [...symbols].sort().join(',')] as const,
};

// Standard error handler
const defaultErrorHandler = (error: unknown) => {
  console.error('Query error:', error);
  const errorMessage = error instanceof Error ? error.message : 'An error occurred';
  
  toast({
    title: 'Error',
    description: errorMessage,
    variant: 'destructive',
  });
};

// Account hooks
export function useAccount() {
  return useQuery({
    queryKey: queryKeys.account,
    queryFn: () => alpaca.getAccount(),
    staleTime: 60 * 1000, // 1 minute
  });
}

type Quote = {
  symbol: string;
  price: number;
  timestamp: string;
};

type QuotesMap = {
  [symbol: string]: Quote;
};

export function useQuotes(symbols: string[]) {
  return useQuery<QuotesMap>({
    queryKey: queryKeys.quotes(symbols),
    queryFn: async () => {
      // Don't fetch if there are no symbols to prevent unnecessary requests
      if (symbols.length === 0) {
        return {};
      }
      const { data } = await axios.get(
        `/api/alpaca/quotes?symbols=${symbols.join(',')}`
      );
      return data;
    },
    // Only run the query if the symbols array is not empty
    enabled: symbols.length > 0,
    staleTime: 30 * 1000, // Keep data fresh for 30 seconds
    refetchInterval: 60 * 1000, // Optional: refetch every minute
  });
}


// Position hooks
export function usePositions() {
  const queryClient = useQueryClient();
  
  const query = useQuery({
    queryKey: queryKeys.positions,
    queryFn: () => alpaca.getPositions(),
    staleTime: 30 * 1000, // 30 seconds
  });
  
  const closePositionMutation = useMutation({
    mutationFn: (symbol: string) => alpaca.closePosition(symbol),
    onSuccess: () => {
      // Invalidate positions query to refetch after closing a position
      queryClient.invalidateQueries({ queryKey: queryKeys.positions });
      toast({
        title: 'Position Closed',
        description: 'The position has been closed successfully',
      });
    },
    onError: (error: unknown) => {
      defaultErrorHandler(error);
    },
  });
  
  return {
    ...query,
    closePosition: closePositionMutation.mutate,
    isClosing: closePositionMutation.isPending,
  };
}

// Portfolio history hooks
export function usePortfolioHistory(days: number, timeframe: string) {
  return useQuery({
    queryKey: queryKeys.portfolioHistory(days, timeframe),
    queryFn: () => alpaca.getPortfolioHistory(days, timeframe),
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}

// Watchlists hooks
export function useWatchlists() {
  return useQuery({
    queryKey: queryKeys.watchlists,
    queryFn: () => alpaca.getWatchlists(),
    staleTime: 60 * 60 * 1000, // 1 hour
  });
}

// Orders hooks
export function useOrders(status?: string) {
  const queryClient = useQueryClient();
  
  const query = useQuery({
    queryKey: queryKeys.orders(status),
    queryFn: () => alpaca.getOrders({ status }),
    staleTime: 30 * 1000, // 30 seconds
  });
  
  const createOrderMutation = useMutation({
    mutationFn: alpaca.createOrder,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.orders() });
      toast({
        title: 'Order Created',
        description: 'The order has been created successfully',
      });
    },
    onError: (error: unknown) => {
      defaultErrorHandler(error);
    },
  });
  
  const cancelOrderMutation = useMutation({
    mutationFn: alpaca.cancelOrder,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.orders() });
      toast({
        title: 'Order Canceled',
        description: 'The order has been canceled successfully',
      });
    },
    onError: (error: unknown) => {
      defaultErrorHandler(error);
    },
  });
  
  const cancelAllOrdersMutation = useMutation({
    mutationFn: alpaca.cancelAllOrders,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.orders() });
      toast({
        title: 'Orders Canceled',
        description: 'All orders have been canceled successfully',
      });
    },
    onError: (error: unknown) => {
      defaultErrorHandler(error);
    },
  });
  
  return {
    ...query,
    createOrder: createOrderMutation.mutate,
    isCreatingOrder: createOrderMutation.isPending,
    cancelOrder: cancelOrderMutation.mutate,
    isCancelingOrder: cancelOrderMutation.isPending,
    cancelAllOrders: cancelAllOrdersMutation.mutate,
    isCancelingAllOrders: cancelAllOrdersMutation.isPending,
  };
}

// Portfolio recommendations hooks
export function usePortfolioRecommendations(params: alpaca.PortfolioRecommendationsParams = {}) {
  return useQuery({
    queryKey: queryKeys.recommendations(params),
    queryFn: () => alpaca.getPortfolioRecommendations(params),
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}

// User hooks
export function useUser() {
  return useQuery({
    queryKey: queryKeys.user,
    queryFn: () => auth.getCurrentUser(),
    staleTime: 5 * 60 * 1000, // 5 minutes
    retry: false, // Don't retry if unauthorized
  });
}

// Auth mutation hooks
export function useAuth() {
  const queryClient = useQueryClient();
  
  const loginMutation = useMutation({
    mutationFn: auth.login,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.user });
      toast({
        title: 'Login Successful',
        description: 'You have been logged in successfully',
      });
    },
    onError: (error: unknown) => {
      defaultErrorHandler(error);
    },
  });
  
  const registerMutation = useMutation({
    mutationFn: auth.register,
    onSuccess: () => {
      toast({
        title: 'Registration Successful',
        description: 'Your account has been created successfully',
      });
    },
    onError: (error: unknown) => {
      defaultErrorHandler(error);
    },
  });
  
  const logoutMutation = useMutation({
    mutationFn: auth.logout,
    onSuccess: () => {
      // Clear all queries on logout
      queryClient.clear();
      toast({
        title: 'Logout Successful',
        description: 'You have been logged out',
      });
    },
    onError: (error: unknown) => {
      defaultErrorHandler(error);
    },
  });
  
  const connectAlpacaMutation = useMutation({
    mutationFn: auth.connectAlpaca,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.user });
      toast({
        title: 'Alpaca Connected',
        description: 'Your Alpaca account has been connected successfully',
      });
    },
    onError: (error: unknown) => {
      defaultErrorHandler(error);
    },
  });
  
  return {
    login: loginMutation.mutate,
    isLoggingIn: loginMutation.isPending,
    register: registerMutation.mutate,
    isRegistering: registerMutation.isPending,
    logout: logoutMutation.mutate,
    isLoggingOut: logoutMutation.isPending,
    connectAlpaca: connectAlpacaMutation.mutate,
    isConnectingAlpaca: connectAlpacaMutation.isPending,
  };
}