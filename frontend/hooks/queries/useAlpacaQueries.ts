// frontend/hooks/queries/useAlpacaQueries.ts
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import * as alpacaApi from '@/lib/api/alpaca';
import * as authApi from '@/lib/api/auth';
import { toast } from '@/hooks/use-toast';

// Query key factory for better organization and type safety
export const queryKeys = {
  account: ['account'] as const,
  positions: ['positions'] as const,
  portfolioHistory: (days: number, timeframe: string) => 
    ['portfolioHistory', days.toString(), timeframe] as const,
  watchlists: ['watchlists'] as const,
  orders: (status?: string) => ['orders', status ?? 'all'] as const,
  recommendations: (params: alpacaApi.PortfolioRecommendationsParams) => 
    ['portfolioRecommendations', JSON.stringify(params)] as const,
  user: ['user'] as const,
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
    queryFn: () => alpacaApi.getAccount(),
    staleTime: 60 * 1000, // 1 minute
  });
}

// Position hooks
export function usePositions() {
  const queryClient = useQueryClient();
  
  const query = useQuery({
    queryKey: queryKeys.positions,
    queryFn: () => alpacaApi.getPositions(),
    staleTime: 30 * 1000, // 30 seconds
  });
  
  const closePositionMutation = useMutation({
    mutationFn: (symbol: string) => alpacaApi.closePosition(symbol),
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
    queryFn: () => alpacaApi.getPortfolioHistory(days, timeframe),
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}

// Watchlists hooks
export function useWatchlists() {
  return useQuery({
    queryKey: queryKeys.watchlists,
    queryFn: () => alpacaApi.getWatchlists(),
    staleTime: 60 * 60 * 1000, // 1 hour
  });
}

// Orders hooks
export function useOrders(status?: string) {
  const queryClient = useQueryClient();
  
  const query = useQuery({
    queryKey: queryKeys.orders(status),
    queryFn: () => alpacaApi.getOrders({ status }),
    staleTime: 30 * 1000, // 30 seconds
  });
  
  const createOrderMutation = useMutation({
    mutationFn: alpacaApi.createOrder,
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
    mutationFn: alpacaApi.cancelOrder,
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
    mutationFn: alpacaApi.cancelAllOrders,
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
export function usePortfolioRecommendations(params: alpacaApi.PortfolioRecommendationsParams = {}) {
  return useQuery({
    queryKey: queryKeys.recommendations(params),
    queryFn: () => alpacaApi.getPortfolioRecommendations(params),
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}

// User hooks
export function useUser() {
  return useQuery({
    queryKey: queryKeys.user,
    queryFn: () => authApi.getCurrentUser(),
    staleTime: 5 * 60 * 1000, // 5 minutes
    retry: false, // Don't retry if unauthorized
  });
}

// Auth mutation hooks
export function useAuth() {
  const queryClient = useQueryClient();
  
  const loginMutation = useMutation({
    mutationFn: authApi.login,
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
    mutationFn: authApi.register,
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
    mutationFn: authApi.logout,
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
    mutationFn: authApi.connectAlpaca,
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