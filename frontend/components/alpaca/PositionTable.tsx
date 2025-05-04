import { useState } from "react";
import { usePositions, useAccount } from "@/hooks/queries/useAlpacaQueries";
import { useQuery } from '@tanstack/react-query';
import VirtualizedTable, { ColDef } from "@/components/ui/custom/VirtualizedTable";
import { Position } from "@/types/alpaca";
import { fmtCurrency, fmtPercent } from "@/lib/utils";
import { Input } from "@/components/ui/input";
import { AlertCircle, ArrowUpDown, RefreshCw, Search } from "lucide-react";
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
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { Skeleton } from "@/components/ui/skeleton";
import { usePortfolio } from '@/app/positions/page';
import axios from 'axios';
import { SortConfig } from "@/hooks/useSortableData";
import { useMarketHours } from "@/hooks/queries/useMarketHours";
import { PortfolioRecommendation, PortfolioRecommendationsResponse } from "@/lib/api/alpaca";
import { ActionButton } from "@/components/ui/custom/ActionButton";
import { OrderDialog } from "@/components/ui/custom/OrderDialog";

export function PositionTable({ count }: { count: number }) {
  // All state hooks at the top
  const [searchQuery, setSearchQuery] = useState("");
  // const [closingPositionSymbol, setClosingPositionSymbol] = useState<string | null>(null);
  
  // Track which symbol's dialog is open instead of a general boolean
  const [activeDialogSymbol, setActiveDialogSymbol] = useState<string | null>(null);
  
  // Sort state
  const [sortConfig, setSortConfig] = useState<SortConfig>({
    key: "market_value", 
    direction: "desc"
  });
  
  // Context and queries
  const { executingOrderSymbol, executeOrder, refreshAllData, lookbackDays } = usePortfolio();
  const { data: positions, isLoading, isError, error, refetch } = usePositions();
  const { data: accountData } = useAccount();
  const { isMarketOpen = true } = useMarketHours() || {};

  // Recommendations query
  const { data: recommendationsData } = useQuery<PortfolioRecommendationsResponse>({
    queryKey: ['portfolioRecommendations', lookbackDays],
    queryFn: async () => {
      const response = await axios.get('/api/alpaca/portfolio/recommendations', {
        params: {
          lookback_days: lookbackDays,
          min_change_percent: 0.01,
          cash_reserve_percent: 0.05
        }
      });
      return response.data;
    },
    enabled: true,
    refetchOnWindowFocus: false,
  });

  // Early return handlers for loading, error, and empty states
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
        <p className="text-muted-foreground mb-4">You dont have any open positions in your portfolio.</p>
        <Button onClick={() => refetch()} variant="outline">
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>
    );
  }

  // Create recommendations map
  const recommendationsMap = new Map<string, PortfolioRecommendation>();
  if (recommendationsData?.recommendations) {
    recommendationsData.recommendations.forEach((rec) => {
      recommendationsMap.set(rec.symbol, rec);
    });
  }

  // Filter positions based on search
  const filteredPositions = positions.filter(p => 
    p.symbol.toLowerCase().includes(searchQuery.toLowerCase())
  );
  
  // Request sort function
  const requestSort = (key: string) => {
    setSortConfig(currentConfig => {
      if (currentConfig.key === key) {
        return {
          key,
          direction: currentConfig.direction === 'asc' ? 'desc' : 'asc',
        };
      }
      return { key, direction: 'asc' };
    });
  };

  // Position comparison function for sorting
  const comparePositions = (a: Position, b: Position, config: SortConfig) => {
    const { key, direction } = config;
    let valueA: string | number, valueB: string | number;
    switch (key) {
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
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        valueA = key in a ? (a as any)[key] : null;
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        valueB = key in b ? (b as any)[key] : null;
    }
    
    const compareResult = valueA > valueB ? 1 : valueA < valueB ? -1 : 0;
    return direction === "asc" ? compareResult : -compareResult;
  };
  
  // Sort the positions manually
  const sortedPositions = [...filteredPositions].sort((a, b) => comparePositions(a, b, sortConfig));

  // Order execution handler - fixed to close the proper dialog
  const handleExecuteOrder = (symbol: string, action: 'Buy' | 'Sell', quantity: number) => {
    // Close the dialog first
    setActiveDialogSymbol(null);
    
    // Then execute the order after a small delay
    setTimeout(() => {
      executeOrder(symbol, action, quantity).catch(err => {
        toast({
          title: "Order Failed",
          description: err instanceof Error ? err.message : "Unknown error occurred",
          variant: "destructive",
        });
      });
    }, 50);
  };

  // Create a sortable header
  const SortableHeader = ({ column, label }: { column: string, label: string }) => (
    <Button
      variant="ghost"
      size="sm"
      className="px-1 py-0 h-auto hover:bg-transparent focus:bg-transparent"
      onClick={() => requestSort(column)}
    >
      {label}
      <ArrowUpDown className={`ml-1 h-3 w-3 ${sortConfig.key === column ? 'opacity-100' : 'opacity-30'}`} />
    </Button>
  );

  // Calculate portfolio summary
  const totalMarketValue = positions.reduce((sum, p) => sum + parseFloat(p.market_value), 0);
  const totalCostBasis = positions.reduce((sum, p) => sum + (parseFloat(p.avg_entry_price) * parseFloat(p.qty)), 0);
  const totalPL = positions.reduce((sum, p) => sum + parseFloat(p.unrealized_pl), 0);
  const totalPLPercent = (totalPL / totalCostBasis) * 100;

  // Render recommendation with fixed dialog behavior
  const renderRecommendation = (p: Position) => {
    const rec = recommendationsMap.get(p.symbol);
    if (!rec) return <span className="text-muted-foreground">—</span>;
    
    const currentPrice = p.current_price 
      ? parseFloat(p.current_price) 
      : (parseFloat(p.market_value) / parseFloat(p.qty));
      
    // Determine if this specific symbol's dialog should be open
    const isThisDialogOpen = activeDialogSymbol === p.symbol;
    const isExecuting = executingOrderSymbol === p.symbol;
    
    return (
      // Each AlertDialog needs its own AlertDialogTrigger inside it
      <AlertDialog open={isThisDialogOpen} onOpenChange={(open) => {
        if (open) {
          setActiveDialogSymbol(p.symbol);
        } else {
          setActiveDialogSymbol(null);
        }
      }}>
        <AlertDialogTrigger asChild>
          <div>
            <ActionButton 
              action={rec.action}
              quantity={rec.quantity}
              price={currentPrice}
              // symbol={p.symbol}
              isExecuting={isExecuting}
            />
          </div>
        </AlertDialogTrigger>
        
        {/* Now OrderDialog only renders the content, not the full AlertDialog */}
        <OrderDialog 
          symbol={p.symbol}
          action={rec.action}
          quantity={rec.quantity}
          price={currentPrice}
          isOpen={isThisDialogOpen}
          onOpenChange={(open) => {
            if (!open) setActiveDialogSymbol(null);
          }}
          onExecute={handleExecuteOrder}
          isExecuting={isExecuting}
          availableCash={accountData ? parseFloat(accountData.cash) : undefined}
        />
      </AlertDialog>
    );
  };

  // Table definition
  const tableDef: ColDef<Position>[] = [
    // Column definitions - header, alignment, cell content, etc.
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
        return rec ? `${rec.action}` : "—";
      },
      align: "right",
      render: renderRecommendation
    },
    {
      label: "Actions",
      value: (_p: Position) => "",
      align: "right",
      render: (p: Position) => (
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
            {/* Additional dropdown items */}
          </DropdownMenuContent>
        </DropdownMenu>
      )
    }
  ];
  
  return (
    <>
      {/* Market warning banner */}
      {!isMarketOpen && (
        <div className="bg-amber-100 dark:bg-amber-900/30 rounded-md p-3 mb-4">
          <p className="text-amber-600">Market is currently closed</p>
        </div>
      )}
      
      <div className="flex justify-between items-center mb-4">
        <div className="text-sm">
          <span className="mr-4">Total Value: {fmtCurrency(totalMarketValue)}</span>
          <span className="mr-4">P/L: {fmtCurrency(totalPL)} ({fmtPercent(totalPLPercent/100)})</span>
          <span className="mr-4">Cash: {accountData ? fmtCurrency(parseFloat(accountData.cash)) : 'Loading...'}</span>
          <span className="mr-4">Day Trades: {accountData ? accountData.daytrade_count : ''}</span>
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