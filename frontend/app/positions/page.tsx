"use client";
import { createContext, useContext, useState } from "react";
import { PositionTable } from "@/components/alpaca/PositionTable";
import { OrderForm } from "@/components/alpaca/OrderForm";
import { PortfolioRecommendations } from "@/components/alpaca/PortfolioRecommendations";
import { useQueryClient } from "@tanstack/react-query";
import { toast } from "@/hooks/use-toast";
import axios from "axios";
import { alpaca } from '@/lib/api'; // Import the API client

// Create a shared context for portfolio data management
export interface PortfolioContextType {
  executingOrderSymbol: string | null;
  setExecutingOrderSymbol: (symbol: string | null) => void;
  executeOrder: (symbol: string, action: 'Buy' | 'Sell', quantity: number) => Promise<void>;
  refreshAllData: () => Promise<void>;
}

export const PortfolioContext = createContext<PortfolioContextType | undefined>(undefined);

// Custom hook for accessing the portfolio context
export function usePortfolio() {
  const context = useContext(PortfolioContext);
  if (!context) {
    throw new Error("usePortfolio must be used within a PortfolioProvider");
  }
  return context;
}

export default function PositionsPage() {
  const [executingOrderSymbol, setExecutingOrderSymbol] = useState<string | null>(null);
  const queryClient = useQueryClient();

  // Enhanced order execution function with better error handling
  const executeOrder = async (symbol: string, action: 'Buy' | 'Sell', quantity: number) => {
    try {
      setExecutingOrderSymbol(symbol);
      
      // First check account to verify sufficient funds for buy orders
      if (action === 'Buy') {
        try {
          const account = await axios.get('/api/alpaca/account');
          const buyingPower = parseFloat(account.data.buying_power);
          
          // Get current price and calculate estimated cost
          const quote = await axios.get(`/api/alpaca/quote/${symbol}`);
          const currentPrice = quote.data.price || 0;
          const estimatedCost = currentPrice * quantity;
          
          if (estimatedCost > buyingPower * 0.99) { // 1% buffer for price changes
            toast({
              title: "Insufficient Funds",
              description: `This order requires ~$${estimatedCost.toFixed(2)} but you only have $${buyingPower.toFixed(2)} available.`,
              variant: "destructive",
            });
            return; // Exit early
          }
        } catch (error) {
          console.log("Error checking account balance:", error);
          // Continue with order - we'll let the API handle the rejection if needed
        }
      }
      
      toast({
        title: `${action} Order Submitted`,
        description: `${action === 'Buy' ? 'Buying' : 'Selling'} ${quantity} shares of ${symbol}...`,
      });
      
      try {
        await axios.post('/api/alpaca/orders', {
          symbol,
          qty: quantity,
          side: action.toLowerCase(),
          type: 'market',
          time_in_force: 'day'
        });
        
        // Clear backend cache
        try {
          await axios.post('http://localhost:8001/api/portfolio/clear-cache');
        } catch (cacheError) {
          // Non-critical error, just log it
          console.log("Cache clear failed:", cacheError);
        }
        
        toast({
          title: "Order Placed Successfully",
          description: `Your ${action.toLowerCase()} order for ${quantity} shares of ${symbol} has been submitted.`,
          variant: "default",
        });
        
        // Refresh all data
        await refreshAllData();
        
      } catch (error) {
        handleOrderError(error, symbol, action, quantity);
      }
      
    } catch (err) {
      // This should only catch unexpected errors outside the main try/catch
      console.error("Unexpected error in order execution:", err);
      toast({
        title: "System Error",
        description: "An unexpected error occurred. Our team has been notified.",
        variant: "destructive",
      });
    } finally {
      setExecutingOrderSymbol(null);
    }
  };
  
  // Enhanced error handler for order execution
  const handleOrderError = (error: unknown, symbol: string, action: string, quantity: number) => {
    console.error("Order execution error:", error);
    
    // Extract error details
    let errorTitle = "Order Failed";
    let errorMessage = "Failed to place order. Please try again.";
    let errorDetails = "";
    
    if (axios.isAxiosError(error) && error.response) {
      const data = error.response.data;
      
      // Try to extract alpaca-specific errors
      if (data.details && typeof data.details === 'string') {
        errorMessage = data.details;
        
        // Check for common error types and create user-friendly messages
        if (errorMessage.includes("insufficient buying power")) {
          errorTitle = "Insufficient Funds";
          const matches = errorMessage.match(/buying_power":"([^"]+)"/);
          const buyingPower = matches ? matches[1] : "unknown";
          errorMessage = `You don't have enough buying power for this transaction. Available: $${buyingPower}`;
        }
        
        // More specific error handling for other common cases
        else if (errorMessage.includes("position is not found")) {
          errorMessage = `You don't currently own any shares of ${symbol}.`;
        }
      } else if (data.error) {
        errorMessage = data.error;
        if (data.details) errorDetails = data.details;
      }
    } else if (error instanceof Error) {
      errorMessage = error.message;
    }
    
    // Show toast with detailed error info
    toast({
      title: errorTitle,
      description: (
        <div>
          <p>{errorMessage}</p>
          {errorDetails && <p className="text-xs mt-2 text-muted-foreground">{errorDetails}</p>}
        </div>
      ),
      variant: "destructive",
    });
    
    // Additional logging for debugging
    console.warn(`Order failed: ${action} ${quantity} ${symbol}`, {
      errorTitle,
      errorMessage,
      errorDetails,
      originalError: error
    });
  };

  // Shared function to refresh all portfolio data
  const refreshAllData = async () => {
    // Remove all cached data
    queryClient.removeQueries({ queryKey: ['portfolioRecommendations'] });
    queryClient.removeQueries({ queryKey: ['positions'] });
    queryClient.removeQueries({ queryKey: ['account'] });
    
    // Refetch fresh data
    await Promise.all([
      queryClient.refetchQueries({ queryKey: ['positions'] }),
      queryClient.refetchQueries({ queryKey: ['account'] }),
      queryClient.refetchQueries({ queryKey: ['portfolioRecommendations'] }),
    ]);
    
    // Setup additional refresh polling to ensure data consistency
    const pollTimes = [2000, 5000];
    for (const delay of pollTimes) {
      setTimeout(() => {
        queryClient.refetchQueries({ queryKey: ['positions'] });
        queryClient.refetchQueries({ queryKey: ['account'] });
        queryClient.refetchQueries({ queryKey: ['portfolioRecommendations'] });
      }, delay);
    }
  };

  // Create the context value
  const contextValue: PortfolioContextType = {
    executingOrderSymbol,
    setExecutingOrderSymbol,
    executeOrder,
    refreshAllData,
  };

  return (
    <PortfolioContext.Provider value={contextValue}>
      <div className="flex">
        <div className="flex-1 p-6 mx-16">
          <div className="flex justify-between items-center mb-6">
            <h1 className="text-3xl font-bold">Open Positions</h1>
            <OrderForm />
          </div>
          <PositionTable count={15} />
          <div className="mb-8 pt-4">
            <PortfolioRecommendations />
          </div>
        </div>
      </div>
    </PortfolioContext.Provider>
  );
}