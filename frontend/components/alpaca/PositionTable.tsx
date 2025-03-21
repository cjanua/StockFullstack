import { useState } from "react";
import { usePositions, useAccount } from "@/hooks/queries/useAlpacaQueries";
import { useQuery, useQueryClient } from '@tanstack/react-query';
import VirtualizedTable, { ColDef } from "@/components/ui/custom/VirtualizedTable";
import { Position } from "@/types/alpaca";
import { fmtCurrency, fmtPercent, fmtCurrencyPrecise, fmtShares } from "@/lib/utils";
import { Input } from "@/components/ui/input";
import { AlertCircle, ArrowUpDown, Loader2, RefreshCw, Search, TrendingUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { toast } from "@/hooks/use-toast";
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
} from "@/components/ui/alert-dialog";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { usePortfolio } from '@/app/positions/page';
import axios from 'axios';

type SortDirection = "asc" | "desc";

interface SortState {
  column: string;
  direction: SortDirection;
}

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
}

export function PositionTable({ count }: { count: number }) {
  // Use shared portfolio context
  const { executingOrderSymbol, setExecutingOrderSymbol, executeOrder, refreshAllData } = usePortfolio();
  
  // Existing queries
  const { 
    data: positions, 
    isLoading, 
    isError, 
    error, 
    refetch, 
    closePosition, 
    isClosing 
  } = usePositions();

  const { data: accountData } = useAccount();

  const queryClient = useQueryClient();

  // Use existing recommendations query (it will be refreshed via the shared context)
  const { data: recommendationsData } = useQuery<RecommendationResponse>({
    queryKey: ['portfolioRecommendations'],
    queryFn: async () => {
      console.log('Fetching fresh recommendations');
      const response = await axios.get('/api/alpaca/portfolio/recommendations', {
        params: {
          lookback_days: 365,
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

  const [searchQuery, setSearchQuery] = useState("");
  const [sortState, setSortState] = useState<SortState>({ 
    column: "market_value", 
    direction: "desc" 
  });
  const [closingPositionSymbol, setClosingPositionSymbol] = useState<string | null>(null);
  
  if (isLoading) {
    return (
      <div className="space-y-4">
        <div className="flex justify-between">
          <Skeleton className="h-10 w-64" />
          <Skeleton className="h-10 w-32" />
        </div>
        <div className="border rounded-md p-4">
          <Skeleton className="h-8 w-full mb-4" />
          {Array(count).fill(0).map((_, i) => (
            <Skeleton key={i} className="h-10 w-full mb-2" />
          ))}
        </div>
      </div>
    );
  }
  
  if (isError) {
    return (
      <div className="rounded-md bg-destructive/10 p-4 flex items-start">
        <AlertCircle className="h-5 w-5 text-destructive mr-2 mt-0.5" />
        <div>
          <h3 className="font-medium text-destructive">Error loading positions</h3>
          <p className="text-sm text-destructive/80">
            {error instanceof Error ? error.message : "An unknown error occurred. Please try again."}
          </p>
          <Button 
            variant="outline" 
            size="sm" 
            onClick={() => refetch()} 
            className="mt-2"
          >
            Retry
          </Button>
        </div>
      </div>
    );
  }
  
  if (!positions || positions.length === 0) {
    return (
      <div className="border rounded-md p-8 text-center">
        <h3 className="font-medium text-lg mb-2">No positions found</h3>
        <p className="text-muted-foreground mb-4">You don't have any open positions in your portfolio.</p>
        <Button onClick={() => refetch()} variant="outline">
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>
    );
  }

  // Create a map of recommendations by symbol for quick lookup
  const recommendationsMap = new Map<string, PortfolioRecommendation>();
  recommendationsData?.recommendations.forEach(rec => {
    recommendationsMap.set(rec.symbol, rec);
  });

  // Filter positions based on search query
  const filteredPositions = positions.filter(p => 
    p.symbol.toLowerCase().includes(searchQuery.toLowerCase())
  );
  
  // Sort positions based on current sort state
  const sortedPositions = [...filteredPositions].sort((a, b) => {
    const { column, direction } = sortState;
    let valueA, valueB;
    
    // Handle different column types
    switch (column) {
      case "symbol":
        valueA = a.symbol;
        valueB = b.symbol;
        break;
      case "cost_basis":
        valueA = parseFloat(a.avg_entry_price) * parseFloat(a.qty);
        valueB = parseFloat(b.avg_entry_price) * parseFloat(b.qty);
        break;
      case "market_value":
        valueA = parseFloat(a.market_value);
        valueB = parseFloat(b.market_value);
        break;
      case "change_today":
        valueA = parseFloat(a.change_today);
        valueB = parseFloat(b.change_today);
        break;
      case "unrealized_intraday_pl":
        valueA = parseFloat(a.unrealized_intraday_pl);
        valueB = parseFloat(b.unrealized_intraday_pl);
        break;
      case "unrealized_plpc":
        valueA = parseFloat(a.unrealized_plpc);
        valueB = parseFloat(b.unrealized_plpc);
        break;
      case "unrealized_pl":
        valueA = parseFloat(a.unrealized_pl);
        valueB = parseFloat(b.unrealized_pl);
        break;
      case "recommendation":
        const recA = recommendationsMap.get(a.symbol);
        const recB = recommendationsMap.get(b.symbol);
        
        // Get correct per-share price
        const priceA = a.current_price 
          ? parseFloat(a.current_price) 
          : (parseFloat(a.market_value) / parseFloat(a.qty));
        const priceB = b.current_price 
          ? parseFloat(b.current_price) 
          : (parseFloat(b.market_value) / parseFloat(b.qty));
        
        // Calculate the absolute monetary values
        const moneyValueA = recA ? Math.abs(recA.quantity * priceA) : 0;
        const moneyValueB = recB ? Math.abs(recB.quantity * priceB) : 0;
        
        // Store the action type (Buy=1, Sell=-1, None=0) for secondary sorting
        const actionTypeA = recA ? (recA.action === 'Buy' ? 1 : -1) : 0;
        const actionTypeB = recB ? (recB.action === 'Buy' ? 1 : -1) : 0;
        
        // First compare by absolute monetary value
        if (Math.abs(moneyValueA - moneyValueB) > 0.0001) {
          valueA = moneyValueA;
          valueB = moneyValueB;
        } else {
          // If monetary values are approximately equal, sort by action type (Buy first, then Sell)
          valueA = actionTypeA;
          valueB = actionTypeB;
        }
        break;
      default:
        valueA = 0;
        valueB = 0;
    }
    
    // Determine sort order
    const compareResult = valueA > valueB ? 1 : valueA < valueB ? -1 : 0;
    return direction === "asc" ? compareResult : -compareResult;
  });
  
  // Function to toggle sort state
  const toggleSort = (column: string) => {
    setSortState(prev => {
      if (prev.column === column) {
        // Toggle direction if same column
        return { column, direction: prev.direction === "asc" ? "desc" : "asc" };
      }
      // Default to descending for new column
      return { column, direction: "desc" };
    });
  };
  
  // Updated function to use shared context for executing orders
  const handleRecommendedAction = (symbol: string, action: 'Buy' | 'Sell', quantity: number) => {
    executeOrder(symbol, action, quantity);
  };

  // Updated function to close a position with improved handling
  const handleClosePosition = async (symbol: string) => {
    setClosingPositionSymbol(symbol);
    
    try {
      // Close the position
      await closePosition(symbol);
      
      // Record the closure in the backend to prevent immediate re-recommendation
      try {
        await axios.post(`http://localhost:8001/api/portfolio/record-position-close/${symbol}`);
        console.log(`Recorded closure of ${symbol}`);
      } catch (recordError) {
        console.error("Failed to record position closure:", recordError);
      }
      
      // Show immediate feedback
      toast({
        title: "Position Closed",
        description: `Closed ${symbol} position successfully.`,
      });
      
      // Use shared refresh function
      await refreshAllData();
      
    } catch (error) {
      console.error(`Error closing position ${symbol}:`, error);
      toast({
        title: "Error",
        description: `Failed to close position: ${error instanceof Error ? error.message : "Unknown error"}`,
        variant: "destructive",
      });
    } finally {
      setClosingPositionSymbol(null);
    }
  };

  // Create a sortable header
  const SortableHeader = ({ column, label }: { column: string, label: string }) => (
    <Button
      variant="ghost"
      size="sm"
      className="px-1 py-0 h-auto hover:bg-transparent focus:bg-transparent"
      onClick={() => toggleSort(column)}
    >
      {label}
      <ArrowUpDown className={`ml-1 h-3 w-3 ${sortState.column === column ? 'opacity-100' : 'opacity-30'}`} />
    </Button>
  );

  // Calculate portfolio summary
  const totalMarketValue = positions.reduce((sum, p) => sum + parseFloat(p.market_value), 0);
  const totalCostBasis = positions.reduce((sum, p) => sum + (parseFloat(p.avg_entry_price) * parseFloat(p.qty)), 0);
  const totalPL = positions.reduce((sum, p) => sum + parseFloat(p.unrealized_pl), 0);
  const totalPLPercent = (totalPL / totalCostBasis) * 100;

  const handleShowRecommendation = (p: Position, rec: PortfolioRecommendation) => {
    // Get accurate current price
    const currentPrice = p.current_price 
      ? parseFloat(p.current_price) 
      : (parseFloat(p.market_value) / parseFloat(p.qty));
    
    // Calculate actual values using current price  
    const estimatedValue = rec.quantity * currentPrice;
    const actionText = rec.action === 'Buy' ? 'purchase' : 'sale';
    
    // Show confirmation with accurate price info
    return (
      <AlertDialog>
        <AlertDialogTrigger asChild>
          <Badge 
            variant={rec.action === 'Buy' ? "outline" : "secondary"} 
            className={`
              ${rec.action === 'Buy' ? 'border-green-500 text-green-500' : 'border-red-500 text-red-500'}
              ml-auto inline-flex w-24 justify-end cursor-pointer hover:opacity-80
            `}
          >
            <div className="flex flex-col items-end group">
              <span className="group-hover:hidden">{fmtCurrency(estimatedValue)}</span>
              <span className="hidden group-hover:block">{rec.action} {fmtShares(rec.quantity)}</span>
            </div>
          </Badge>
        </AlertDialogTrigger>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{rec.action} {p.symbol} Shares</AlertDialogTitle>
            <AlertDialogDescription>
              This will create a market order to {rec.action.toLowerCase()} {fmtShares(rec.quantity)} shares
              of {p.symbol}.
            </AlertDialogDescription>
            
            <div className="mt-4 p-3 bg-muted rounded-md">
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div>Symbol:</div>
                <div className="font-medium text-right">{p.symbol}</div>
                
                <div>Current Price:</div>
                <div className="font-medium text-right">{fmtCurrencyPrecise(currentPrice)}</div>
                
                <div>Quantity:</div>
                <div className="font-medium text-right">{fmtShares(rec.quantity)} shares</div>
                
                <div className="font-medium">Estimated {actionText}:</div>
                <div className={`font-medium text-right ${rec.action === 'Buy' ? 'text-green-500' : 'text-red-500'}`}>
                  {fmtCurrency(estimatedValue)}
                </div>
                
                {rec.action === 'Buy' && accountData && (
                  <>
                    <div>Available Cash:</div>
                    <div className="font-medium text-right">
                      {fmtCurrency(parseFloat(accountData.cash))}
                    </div>
                    {parseFloat(accountData.cash) < estimatedValue && (
                      <div className="col-span-2 text-red-500 text-xs">
                        Warning: This purchase may exceed your available cash. 
                        The order might be rejected.
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => executeOrder(p.symbol, rec.action, rec.quantity)}
              disabled={executingOrderSymbol === p.symbol}
              className={rec.action === 'Buy' ? "bg-green-500 hover:bg-green-600" : "bg-red-500 hover:bg-red-600"}
            >
              {executingOrderSymbol === p.symbol ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Submitting...
                </>
              ) : (
                `${rec.action} Shares`
              )}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    );
  };

  const tableDef: ColDef<Position>[] = [
    {
      label: <SortableHeader column="symbol" label="Symbol" />,
      value: (p: Position) => p.symbol,
      align: "left",
    },
    {
      label: <SortableHeader column="cost_basis" label="Cost" />,
      value: (p: Position) => fmtCurrency(
        parseFloat(p.avg_entry_price) * parseFloat(p.qty),
      ),
      align: "right",
    },
    {
      label: <SortableHeader column="market_value" label="Current" />,
      value: (p: Position) => fmtCurrency(
        parseFloat(p.market_value)
      ),
      align: "right",
    },
    {
      label: <SortableHeader column="change_today" label="% TDY" />,
      value: (p: Position) => `${fmtPercent(parseFloat(p.change_today))}`,
      align: "right",
      className: (p: Position) => {
        const value = parseFloat(p.change_today);
        if (value > 0) return "text-green-400";
        if (value < 0) return "text-red-400";
        return "text-gray-400";
      }
    },
    {
      label: <SortableHeader column="unrealized_intraday_pl" label="TDY $ PL" />,
      value: (p: Position) => fmtCurrency(
        parseFloat(p.unrealized_intraday_pl),
      ),
      align: "right",
      className: (p: Position) => {
        const value = parseFloat(p.unrealized_intraday_pl);
        if (value > 0) return "text-green-400";
        if (value < 0) return "text-red-400";
        return "text-gray-400";
      }
    },
    {
      label: <SortableHeader column="unrealized_plpc" label="Net PL %" />,
      value: (p: Position) => fmtPercent(parseFloat(p.unrealized_plpc)),
      align: "right",
      className: (p: Position) => {
        const value = parseFloat(p.unrealized_plpc);
        if (value > 0) return "text-green-400";
        if (value < 0) return "text-red-400";
        return "text-gray-400";
      }
    },
    {
      label: <SortableHeader column="unrealized_pl" label="Net PL $" />,
      value: (p: Position) => fmtCurrency(parseFloat(p.unrealized_pl)),
      align: "right",
      className: (p: Position) => {
        const value = parseFloat(p.unrealized_pl);
        if (value > 0) return "text-green-400";
        if (value < 0) return "text-red-400";
        return "text-gray-400";
      }
    },
    {
      label: <SortableHeader column="recommendation" label="Recommended Action" />,
      value: (p: Position) => {
        const rec = recommendationsMap.get(p.symbol);
        if (!rec) return "—";
        
        // Calculate accurate per-share price
        const currentPrice = p.current_price 
          ? parseFloat(p.current_price) 
          : parseFloat(p.market_value) / parseFloat(p.qty);
          
        // Calculate monetary value of recommendation
        const moneyValue = rec.quantity * currentPrice;
        
        return `${rec.action} ${fmtShares(rec.quantity)} (${fmtCurrency(moneyValue)})`;
      },
      align: "right",
      render: (p: Position) => {
        const rec = recommendationsMap.get(p.symbol);
        if (!rec) return <span className="text-muted-foreground">—</span>;
        
        // Calculate accurate per-share price
        const currentPrice = p.current_price 
          ? parseFloat(p.current_price) 
          : parseFloat(p.market_value) / parseFloat(p.qty);
          
        // Calculate monetary value of recommendation
        const moneyValue = rec.quantity * currentPrice;
        const estimatedValue = rec.quantity * currentPrice;
        
        return handleShowRecommendation(p, rec);
      }
    },
    {
      label: "Actions",
      value: (_p: Position) => "",
      align: "right",
      render: (p: Position) => {
        const rec = recommendationsMap.get(p.symbol);
        
        // Calculate accurate per-share price
        const currentPrice = p.current_price 
          ? parseFloat(p.current_price) 
          : parseFloat(p.market_value) / parseFloat(p.qty);
    
        return (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                <span className="text-sm">...</span>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuLabel>Actions</DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={() => window.open(`https://finance.yahoo.com/quote/${p.symbol}`, '_blank')}>
                View Details
              </DropdownMenuItem>
              {/* Removed the Recommended Action menu item as it's now accessible via the badge */}
              <AlertDialog>
                <AlertDialogTrigger asChild>
                  <DropdownMenuItem 
                    className="text-red-500" 
                    onSelect={(e) => e.preventDefault()} // Prevent dropdown from closing
                  >
                    Close Position
                  </DropdownMenuItem>
                </AlertDialogTrigger>
                <AlertDialogContent>
                  <AlertDialogHeader>
                    <AlertDialogTitle>Close {p.symbol} Position</AlertDialogTitle>
                    <AlertDialogDescription>
                      This will create a market order to {p.side === 'long' ? 'sell' : 'buy'} your entire position of {fmtShares(p.qty)} shares.
                    </AlertDialogDescription>
                    
                    <div className="mt-4 p-3 bg-muted rounded-md">
                      <div className="grid grid-cols-2 gap-2 text-sm">
                        <div>Symbol:</div>
                        <div className="font-medium text-right">{p.symbol}</div>
                        
                        <div>Current Price:</div>
                        <div className="font-medium text-right">{fmtCurrencyPrecise(currentPrice)}</div>
                        
                        <div>Quantity:</div>
                        <div className="font-medium text-right">{fmtShares(p.qty)} shares</div>
                        
                        <div className="font-medium">Market Value:</div>
                        <div className="font-medium text-right text-red-500">
                          {fmtCurrency(parseFloat(p.market_value))}
                        </div>
                      </div>
                    </div>
                  </AlertDialogHeader>
                  <AlertDialogFooter>
                    <AlertDialogCancel>Cancel</AlertDialogCancel>
                    <AlertDialogAction
                      onClick={() => handleClosePosition(p.symbol)}
                      disabled={closingPositionSymbol === p.symbol || isClosing}
                      className="bg-red-500 hover:bg-red-600"
                    >
                      {closingPositionSymbol === p.symbol ? (
                        <>
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                          Closing...
                        </>
                      ) : (
                        'Close Position'
                      )}
                    </AlertDialogAction>
                  </AlertDialogFooter>
                </AlertDialogContent>
              </AlertDialog>
            </DropdownMenuContent>
          </DropdownMenu>
        );
      }
    }
  ];
  
  return (
    <>
      <div className="flex justify-between items-center mb-4">
        <div className="text-sm">
          <span className="mr-4">Total Value: <span className="font-medium">{fmtCurrency(totalMarketValue)}</span></span>
          <span className="mr-4">P/L: <span className={`font-medium ${
            totalPL > 0 ? 'text-green-400' : 
            totalPL < 0 ? 'text-red-400' : 
            'text-gray-400'
          }`}>
            {fmtCurrency(totalPL)} ({fmtPercent(totalPLPercent/100)})
          </span></span>
          <span className="mr-4">Cash: <span className="font-medium">
            {accountData ? fmtCurrency(parseFloat(accountData.cash)) : 'Loading...'}
          </span></span>
          <span className="mr-4">Day Trades: <span className="font-medium">
            {accountData ? accountData.daytrade_count : ''}
          </span></span>
        </div>
        <div className="flex items-center gap-4">
          <div className="relative w-64">
            <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search symbol..."
              className="pl-8"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>
          {/* Enhanced refresh button with more aggressive cache clearing */}
          <Button 
            variant="outline" 
            size="sm" 
            onClick={refreshAllData}
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
        </div>
      </div>
      
      <VirtualizedTable 
        items={sortedPositions} 
        count={Math.min(count, sortedPositions.length)} 
        tableDef={tableDef}
      />
    </>
  );
}