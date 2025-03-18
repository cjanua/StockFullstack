// frontend/components/alpaca/PositionTable.tsx
import { usePositions } from "@/hooks/alpaca/usePositions";
import VirtualizedTable, { ColDef } from "@/components/ui/custom/VirtualizedTable";
import { Position } from "@/lib/alpaca";
import { fmtCurrency, fmtPercent } from "@/lib/utils";
import { useState } from "react";
import { Input } from "@/components/ui/input";
import { AlertCircle, ArrowUpDown, Loader2, Search } from "lucide-react";
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

type SortDirection = "asc" | "desc";

interface SortState {
  column: string;
  direction: SortDirection;
}

export function PositionTable({ count }: { count: number }) {
  const { positions, isLoading, isError, error, mutate } = usePositions();
  const [searchQuery, setSearchQuery] = useState("");
  const [sortState, setSortState] = useState<SortState>({ 
    column: "market_value", 
    direction: "desc" 
  });
  const [closingPosition, setClosingPosition] = useState<string | null>(null);
  
  if (isLoading) return <div>Loading positions data...</div>;
  if (isError) return error.fallback;
  if (!positions) return <div>No positions data available</div>;

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
  const handleClosePosition = async (symbol: string) => {
    try {
      setClosingPosition(symbol);
      
      // Find position to get side information
      const position = positions.find(p => p.symbol === symbol);
      if (!position) {
        throw new Error(`Position for ${symbol} not found`);
      }
      
      // Determine if long or short position
      const positionSide = position.side === 'long' ? 'long' : 'short';
      
      // Make API call to close position
      const response = await fetch(`/api/alpaca/positions/${symbol}`, {
        method: 'DELETE',
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.details || `Failed to close ${symbol} position`);
      }
      
      // Show success message
      toast({
        title: "Position Closed",
        description: `Successfully closed ${symbol} position with market order`,
      });
      
      // Refresh positions data
      mutate();
    } catch (error) {
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : `Failed to close ${symbol} position`,
        variant: "destructive",
      });
    } finally {
      setClosingPosition(null);
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
                    disabled={closingPosition === p.symbol}
                    className="bg-red-500 hover:bg-red-600"
                  >
                    {closingPosition === p.symbol ? (
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
  
  // Calculate portfolio summary
  const totalMarketValue = positions.reduce((sum, p) => sum + parseFloat(p.market_value), 0);
  const totalCostBasis = positions.reduce((sum, p) => sum + (parseFloat(p.avg_entry_price) * parseFloat(p.qty)), 0);
  const totalPL = positions.reduce((sum, p) => sum + parseFloat(p.unrealized_pl), 0);
  const totalPLPercent = (totalPL / totalCostBasis) * 100;
  
  return (
    <>
      <div className="flex justify-between items-center mb-4">
        <div className="text-sm">
          <span className="mr-4">Total Value: <span className="font-medium">{fmtCurrency(totalMarketValue)}</span></span>
          <span className="mr-4">P/L: <span className={`font-medium ${totalPL > 0 ? 'text-green-400' : 'text-red-400'}`}>
            {fmtCurrency(totalPL)} ({fmtPercent(totalPLPercent/100)})
          </span></span>
        </div>
        <div className="relative w-64">
          <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search symbol..."
            className="pl-8"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
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