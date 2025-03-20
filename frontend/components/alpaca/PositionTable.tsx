// frontend/components/alpaca/PositionTable.tsx
import { useState } from "react";
import { usePositions } from "@/hooks/queries/useAlpacaQueries";
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
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
  // Simple usage of the hook without any custom options
  const { 
    data: positions, 
    isLoading, 
    isError, 
    error, 
    refetch, 
    closePosition, 
    isClosing 
  } = usePositions();

  // Fetch recommendations data
  const { 
    data: recommendationsData, 
    isLoading: isLoadingRecs, 
    refetch: refetchRecs 
  } = useQuery<RecommendationResponse>({
    queryKey: ['portfolioRecommendations'],
    queryFn: async () => {
      const response = await axios.get('/api/alpaca/portfolio/recommendations', {
        params: {
          lookback_days: 365,
          min_change_percent: 0.01,
          cash_reserve_percent: 0.05
        }
      });
      return response.data;
    },
    enabled: true,
    refetchOnWindowFocus: false,
  });

  const [searchQuery, setSearchQuery] = useState("");
  const [sortState, setSortState] = useState<SortState>({ 
    column: "market_value", 
    direction: "desc" 
  });
  const [closingPositionSymbol, setClosingPositionSymbol] = useState<string | null>(null);
  const [executingOrderSymbol, setExecutingOrderSymbol] = useState<string | null>(null);
  
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
  
  // Function to close a position
  const handleClosePosition = (symbol: string) => {
    setClosingPositionSymbol(symbol);
    
    closePosition(symbol);
    
    // Show immediate feedback
    toast({
      title: "Closing Position",
      description: `Closing ${symbol} position...`,
    });
    
    // Clear the closing state after a timeout (this should be handled by the mutation in a real app)
    setTimeout(() => {
      setClosingPositionSymbol(null);
    }, 2000);
  };

  // Function to execute recommended action
  const executeRecommendedAction = async (symbol: string, action: 'Buy' | 'Sell', quantity: number) => {
    try {
      setExecutingOrderSymbol(symbol);
      
      // Show immediate feedback
      toast({
        title: `${action} Order Submitted`,
        description: `${action === 'Buy' ? 'Buying' : 'Selling'} ${fmtShares(quantity)} shares of ${symbol}...`,
      });
      
      // Execute the actual order
      await axios.post('/api/alpaca/orders', {
        symbol,
        qty: quantity,
        side: action.toLowerCase(),
        type: 'market',
        time_in_force: 'day'
      });
      
      // Success feedback
      toast({
        title: "Order Placed Successfully",
        description: `Your ${action.toLowerCase()} order for ${fmtShares(quantity)} shares of ${symbol} has been submitted.`,
        variant: "default",
      });
      
      // Refresh data after the order
      setTimeout(() => {
        refetch();
        refetchRecs();
      }, 1000);
      
    } catch (err) {
      // Error handling
      console.error("Order execution error:", err);
      toast({
        title: "Order Execution Failed",
        description: err instanceof Error ? err.message : "Failed to place order. Please try again.",
        variant: "destructive",
      });
    } finally {
      setExecutingOrderSymbol(null);
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
      className: (p: Position) => parseFloat(p.change_today) > 0 ? "text-green-400" : "text-red-400"
    },
    {
      label: <SortableHeader column="unrealized_intraday_pl" label="TDY $ PL" />,
      value: (p: Position) => fmtCurrency(
        parseFloat(p.unrealized_intraday_pl),
      ),
      align: "right",
      className: (p: Position) => parseFloat(p.unrealized_intraday_pl) > 0 ? "text-green-400" : "text-red-400"
    },
    {
      label: <SortableHeader column="unrealized_plpc" label="Net PL %" />,
      value: (p: Position) => fmtPercent(parseFloat(p.unrealized_plpc)),
      align: "right",
      className: (p: Position) => parseFloat(p.unrealized_plpc) > 0 ? "text-green-400" : "text-red-400"
    },
    {
      label: <SortableHeader column="unrealized_pl" label="Net PL $" />,
      value: (p: Position) => fmtCurrency(parseFloat(p.unrealized_pl)),
      align: "right",
      className: (p: Position) => parseFloat(p.unrealized_pl) > 0 ? "text-green-400" : "text-red-400"
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
                  <span className="group-hover:hidden">{fmtCurrency(moneyValue)}</span>
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
                    
                    <div className="font-medium">Estimated Value:</div>
                    <div className={`font-medium text-right ${rec.action === 'Buy' ? 'text-green-500' : 'text-red-500'}`}>
                      {fmtCurrency(estimatedValue)}
                    </div>
                  </div>
                </div>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Cancel</AlertDialogCancel>
                <AlertDialogAction
                  onClick={() => executeRecommendedAction(p.symbol, rec.action, rec.quantity)}
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
          <span className="mr-4">P/L: <span className={`font-medium ${totalPL > 0 ? 'text-green-400' : 'text-red-400'}`}>
            {fmtCurrency(totalPL)} ({fmtPercent(totalPLPercent/100)})
          </span></span>
          {recommendationsData && (
            <span className="mr-4">Cash: <span className="font-medium">{fmtCurrency(recommendationsData.cash)}</span></span>
          )}
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
          <Button 
            variant="outline" 
            size="sm" 
            onClick={() => {
              refetch();
              refetchRecs();
            }}
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