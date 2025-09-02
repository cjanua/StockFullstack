// New file: app/positions/portfolio-context.tsx
"use client";
import { createContext, useContext } from "react";

// Create a shared context for portfolio data management
export interface PortfolioContextType {
  executingOrderSymbol: string | null;
  setExecutingOrderSymbol: (symbol: string | null) => void;
  executeOrder: (symbol: string, action: 'Buy' | 'Sell', quantity: number) => Promise<void>;
  refreshAllData: () => Promise<void>;
  lookbackDays: number; // Add this to the context
  isProcessingRecommendations: boolean; // Add loading state for recommendations
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