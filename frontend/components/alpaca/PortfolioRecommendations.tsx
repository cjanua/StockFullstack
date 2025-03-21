// components/alpaca/PortfolioRecommendations.tsx
'use client';

import { useState, useEffect } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import axios from 'axios';
import { fmtShares, fmtCurrency, fmtCurrencyPrecise } from '@/lib/utils';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Loader2, TrendingUp, RefreshCw, AlertTriangle } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { toast } from '@/hooks/use-toast';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog';
import { usePortfolio } from '@/app/positions/page';

interface PortfolioRecommendation {
  symbol: string;
  current_shares: number;
  target_shares: number;
  difference: number;
  action: 'Buy' | 'Sell';
  quantity: number;
}

interface RecommendationResponse {
  portfolio_value: number;
  cash: number;
  target_cash: number;
  recommendations: PortfolioRecommendation[];
  processing_time_seconds?: number; // Add this property as optional
}

export function PortfolioRecommendations() {
  // Use shared portfolio context
  const { executingOrderSymbol, executeOrder, refreshAllData, lookbackDays, isProcessingRecommendations } = usePortfolio();
  
  // Add refetch to the destructured values from useQuery
  const { data, isLoading, isError, error, refetch, isFetching } = useQuery<RecommendationResponse>({
    queryKey: ['portfolioRecommendations', lookbackDays], // Use context value
    queryFn: async () => {
      console.log(`Fetching fresh recommendations with ${lookbackDays} days lookback`);
      const response = await axios.get('/api/alpaca/portfolio/recommendations', {
        params: {
          lookback_days: lookbackDays, // Use the shared context value
          min_change_percent: 0.01,
          cash_reserve_percent: 0.05
        },
        headers: {
          'Cache-Control': 'no-cache',
          'Pragma': 'no-cache',
          'X-Refresh-Token': Date.now().toString()
        }
      });
      return response.data;
    },
    enabled: true,
    refetchOnWindowFocus: false,
    staleTime: 10000,
  });
  
  // Use quoted price data for recommendations with safer implementation
  const [symbolPrices, setSymbolPrices] = useState<Record<string, number>>({});

  // Function to fetch current prices with better error handling
  const fetchSymbolPrices = async (symbols: string[]) => {
    try {
      if (!symbols || symbols.length === 0) return;
      
      const uniqueSymbols = [...new Set(symbols)];
      const pricePromises = uniqueSymbols.map(async (symbol) => {
        try {
          const response = await axios.get(`/api/alpaca/quote/${encodeURIComponent(symbol)}`);
          return { 
            symbol, 
            price: response.data?.price || 100 // Fallback price of 100 if not available
          };
        } catch (e) {
          console.error(`Failed to fetch price for ${symbol}:`, e);
          return { symbol, price: 100 }; // Default fallback price
        }
      });
      
      const results = await Promise.all(pricePromises);
      const priceMap = results.reduce((acc, curr) => {
        if (curr && curr.symbol) {
          acc[curr.symbol] = curr.price || 100;
        }
        return acc;
      }, {} as Record<string, number>);
      
      setSymbolPrices(priceMap);
    } catch (e) {
      console.error("Error fetching symbol prices:", e);
      
      // Create fallback prices in case of failure
      if (symbols && symbols.length > 0) {
        const fallbackPrices = symbols.reduce((acc, symbol) => {
          acc[symbol] = 100; // Default price of $100
          return acc;
        }, {} as Record<string, number>);
        
        setSymbolPrices(fallbackPrices);
      }
    }
  };

  // Safer useEffect implementation
  useEffect(() => {
    if (data?.recommendations && Array.isArray(data.recommendations) && data.recommendations.length > 0) {
      const symbols = data.recommendations
        .filter(rec => rec && typeof rec === 'object' && 'symbol' in rec)
        .map(rec => rec.symbol);
      
      if (symbols.length > 0) {
        fetchSymbolPrices(symbols).catch(console.error);
      }
    }
  }, [data?.recommendations]);
  
  // Enhanced execute handler with logging
  const handleExecuteAction = async (symbol: string, action: 'Buy' | 'Sell', quantity: number, event: React.MouseEvent) => {
    // Prevent default behavior and stop propagation
    event.preventDefault();
    event.stopPropagation();
    
    console.log(`Attempting to execute ${action} order for ${quantity} shares of ${symbol}`);
    
    try {
      // Add a callback to run after the dialog is closed
      await executeOrder(symbol, action, quantity);
      
      // Show feedback
      toast({
        title: "Order Submitted",
        description: `${action} order for ${quantity} shares of ${symbol} was submitted`,
      });
    } catch (error) {
      console.error("Failed to execute order:", error);
      toast({
        title: "Order Failed",
        description: "Failed to execute order. Please try again.",
        variant: "destructive",
      });
    }
  };
  
  const buyRecommendations = data?.recommendations.filter(r => r.action === 'Buy') || [];
  const sellRecommendations = data?.recommendations.filter(r => r.action === 'Sell') || [];
  
  // Show better loading states
  const showLoading = isLoading || isFetching || isProcessingRecommendations;
  
  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex justify-between items-center">
          <div>
            <CardTitle className="flex items-center">
              <TrendingUp className="mr-2 h-5 w-5" />
              Portfolio Optimization
            </CardTitle>
            <CardDescription>
              Recommended trades to optimize your portfolio
              {data?.processing_time_seconds && (
                <span className="ml-2 text-xs text-muted-foreground">
                  (Processed in {data.processing_time_seconds}s)
                </span>
              )}
            </CardDescription>
          </div>
          <Button 
            variant="outline" 
            size="sm" 
            onClick={refreshAllData} 
            disabled={showLoading}
          >
            {showLoading ? (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" /> 
            ) : (
              <RefreshCw className="mr-2 h-4 w-4" />
            )}
            Refresh
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {showLoading ? (
          <div className="flex justify-center items-center p-8">
            <div className="flex flex-col items-center">
              <Loader2 className="h-8 w-8 animate-spin text-muted-foreground mb-4" />
              <p className="text-muted-foreground">
                {lookbackDays > 1000 ? "Optimizing portfolio with extended history data..." : "Loading recommendations..."}
              </p>
              {lookbackDays > 3000 && (
                <p className="text-xs text-muted-foreground mt-2">
                  This may take a minute when using 10 years of historical data
                </p>
              )}
            </div>
          </div>
        ) : isError ? (
          <Alert variant="destructive">
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>
              {error instanceof Error 
                ? error.message 
                : "Unable to get portfolio recommendations. Please try again later or check if the portfolio service is running."}
            </AlertDescription>
            <div className="mt-2">
              <Button variant="outline" size="sm" onClick={() => refetch()}>
                Try Again
              </Button>
            </div>
          </Alert>
        ) : (
          <>
            <div className="grid grid-cols-3 gap-4 mb-6">
              <div className="bg-muted/50 p-3 rounded-lg">
                <div className="text-sm text-muted-foreground">Portfolio Value</div>
                <div className="text-xl font-bold mt-1">{fmtCurrency(data?.portfolio_value || 0)}</div>
              </div>
              <div className="bg-muted/50 p-3 rounded-lg">
                <div className="text-sm text-muted-foreground">Current Cash</div>
                <div className="text-xl font-bold mt-1">{fmtCurrency(data?.cash || 0)}</div>
              </div>
              <div className="bg-muted/50 p-3 rounded-lg">
                <div className="text-sm text-muted-foreground">Target Cash</div>
                <div className="text-xl font-bold mt-1">{fmtCurrency(data?.target_cash || 0)}</div>
              </div>
            </div>
            
            <Tabs defaultValue="buy">
              <TabsList className="mb-4">
                <TabsTrigger value="buy">
                  Buy Recommendations
                  {buyRecommendations.length > 0 && (
                    <Badge variant="secondary" className="ml-2">
                      {buyRecommendations.length}
                    </Badge>
                  )}
                </TabsTrigger>
                <TabsTrigger value="sell">
                  Sell Recommendations
                  {sellRecommendations.length > 0 && (
                    <Badge variant="secondary" className="ml-2">
                      {sellRecommendations.length}
                    </Badge>
                  )}
                </TabsTrigger>
              </TabsList>
              
              <TabsContent value="buy">
                {buyRecommendations.length > 0 ? (
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Symbol</TableHead>
                        <TableHead>Current Shares</TableHead>
                        <TableHead>Target Shares</TableHead>
                        <TableHead>Buy Quantity</TableHead>
                        <TableHead className="text-right">Action</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {buyRecommendations.map((rec) => (
                        <TableRow key={rec.symbol}>
                          <TableCell className="font-medium">{rec.symbol}</TableCell>
                          <TableCell>{fmtShares(rec.current_shares)}</TableCell>
                          <TableCell>{fmtShares(rec.target_shares)}</TableCell>
                          <TableCell className="font-bold text-green-600">
                            {fmtShares(rec.quantity)}
                          </TableCell>
                          <TableCell className="text-right">
                            <AlertDialog>
                              <AlertDialogTrigger asChild>
                                <Button 
                                  size="sm" 
                                  variant="outline"
                                  className="border-green-500 text-green-500 hover:bg-green-50"
                                  disabled={executingOrderSymbol === rec.symbol}
                                >
                                  {executingOrderSymbol === rec.symbol ? (
                                    <Loader2 className="h-4 w-4 animate-spin" />
                                  ) : (
                                    'Buy'
                                  )}
                                </Button>
                              </AlertDialogTrigger>
                              <AlertDialogContent>
                                <AlertDialogHeader>
                                  <AlertDialogTitle>Buy {rec.symbol} Shares</AlertDialogTitle>
                                  <AlertDialogDescription>
                                    This will create a market order to buy {fmtShares(rec.quantity)} shares of {rec.symbol}.
                                  </AlertDialogDescription>
                                  
                                  <div className="mt-4 p-3 bg-muted rounded-md">
                                    <div className="grid grid-cols-2 gap-2 text-sm">
                                      <div>Symbol:</div>
                                      <div className="font-medium text-right">{rec.symbol}</div>
                                      
                                      <div>Quantity:</div>
                                      <div className="font-medium text-right">{fmtShares(rec.quantity)} shares</div>
                                      
                                      <div className="font-medium">Estimated Cost:</div>
                                      <div className="font-medium text-right text-green-500">
                                        {symbolPrices && rec.symbol && symbolPrices[rec.symbol] 
                                          ? fmtCurrency(rec.quantity * symbolPrices[rec.symbol])
                                          : `~${fmtCurrency(rec.quantity * 100)}`} {/* Fallback price calculation */}
                                      </div>
                                    </div>
                                  </div>
                                </AlertDialogHeader>
                                <AlertDialogFooter>
                                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                                  <AlertDialogAction
                                    onClick={(e) => handleExecuteAction(rec.symbol, 'Buy', rec.quantity, e)}
                                    className="bg-green-500 hover:bg-green-600"
                                  >
                                    Buy Shares
                                  </AlertDialogAction>
                                </AlertDialogFooter>
                              </AlertDialogContent>
                            </AlertDialog>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                ) : (
                  <div className="text-center py-8 text-muted-foreground">
                    No buy recommendations at this time
                  </div>
                )}
              </TabsContent>
              
              <TabsContent value="sell">
                {sellRecommendations.length > 0 ? (
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Symbol</TableHead>
                        <TableHead>Current Shares</TableHead>
                        <TableHead>Target Shares</TableHead>
                        <TableHead>Sell Quantity</TableHead>
                        <TableHead className="text-right">Action</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {sellRecommendations.map((rec) => (
                        <TableRow key={rec.symbol}>
                          <TableCell className="font-medium">{rec.symbol}</TableCell>
                          <TableCell>{fmtShares(rec.current_shares)}</TableCell>
                          <TableCell>{fmtShares(rec.target_shares)}</TableCell>
                          <TableCell className="font-bold text-red-600">
                            {fmtShares(rec.quantity)}
                          </TableCell>
                          <TableCell className="text-right">
                            <AlertDialog>
                              <AlertDialogTrigger asChild>
                                <Button 
                                  size="sm" 
                                  variant="outline"
                                  className="border-red-500 text-red-500 hover:bg-red-50"
                                  disabled={executingOrderSymbol === rec.symbol}
                                >
                                  {executingOrderSymbol === rec.symbol ? (
                                    <Loader2 className="h-4 w-4 animate-spin" />
                                  ) : (
                                    'Sell'
                                  )}
                                </Button>
                              </AlertDialogTrigger>
                              <AlertDialogContent>
                                <AlertDialogHeader>
                                  <AlertDialogTitle>Sell {rec.symbol} Shares</AlertDialogTitle>
                                  <AlertDialogDescription>
                                    This will create a market order to sell {fmtShares(rec.quantity)} shares of {rec.symbol}.
                                  </AlertDialogDescription>
                                  
                                  <div className="mt-4 p-3 bg-muted rounded-md">
                                    <div className="grid grid-cols-2 gap-2 text-sm">
                                      <div>Symbol:</div>
                                      <div className="font-medium text-right">{rec.symbol}</div>
                                      
                                      <div>Quantity:</div>
                                      <div className="font-medium text-right">{fmtShares(rec.quantity)} shares</div>
                                      
                                      <div className="font-medium">Estimated Value:</div>
                                      <div className="font-medium text-right text-red-500">
                                        {symbolPrices && rec.symbol && symbolPrices[rec.symbol] 
                                          ? fmtCurrency(rec.quantity * symbolPrices[rec.symbol])
                                          : `~${fmtCurrency(rec.quantity * 100)}`} {/* Fallback price calculation */}
                                      </div>
                                    </div>
                                  </div>
                                </AlertDialogHeader>
                                <AlertDialogFooter>
                                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                                  <AlertDialogAction
                                    onClick={(e) => handleExecuteAction(rec.symbol, 'Sell', rec.quantity, e)}
                                    className="bg-red-500 hover:bg-red-600"
                                  >
                                    Sell Shares
                                  </AlertDialogAction>
                                </AlertDialogFooter>
                              </AlertDialogContent>
                            </AlertDialog>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                ) : (
                  <div className="text-center py-8 text-muted-foreground">
                    No sell recommendations at this time
                  </div>
                )}
              </TabsContent>
            </Tabs>
          </>
        )}
      </CardContent>
    </Card>
  );
}