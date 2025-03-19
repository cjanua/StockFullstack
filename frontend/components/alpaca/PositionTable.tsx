// frontend/components/alpaca/PositionTable.tsx
import { useState } from "react";
import { usePositions } from "@/hooks/queries/useAlpacaQueries";
import VirtualizedTable, { ColDef } from "@/components/ui/custom/VirtualizedTable";
import { Position } from "@/types/alpaca";
import { fmtCurrency, fmtPercent } from "@/lib/utils";
import { Input } from "@/components/ui/input";
import { AlertCircle, ArrowUpDown, Loader2, RefreshCw, Search } from "lucide-react";
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

type SortDirection = "asc" | "desc";

interface SortState {
  column: string;
  direction: SortDirection;
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
            <Skeleton key={i} className="h-14 w-full mb-2" />
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
            <DropdownMenuItem>Add Note</DropdownMenuItem>
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
                    This will create a market order to {p.side === 'long' ? 'sell' : 'buy'} your entire position of {p.qty} shares.
                    Are you sure you want to proceed?
                  </AlertDialogDescription>
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
      )
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
          <Button variant="outline" size="sm" onClick={() => refetch()}>
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