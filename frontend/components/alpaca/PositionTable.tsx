import { usePositions } from "@/hooks/alpaca/usePositions";
import VirtualizedTable, { ColDef } from "@/components/ui/custom/VirtualizedTable";
import { Position } from "@/lib/alpaca";
import { fmtCurrency, fmtPercent } from "@/lib/utils";


export function PositionTable({ count }: { count: number }) {
  const { positions, isLoading, isError, error } = usePositions();  

  if (isLoading) return <div>Loading positions data...</div>;
  if (isError) return error.fallback;
  if (!positions) return <div>No positions data available</div>;

  const tableDef: ColDef<Position>[] = [
    {
      label: "Symbol",
      value: (p: Position) => p.symbol,
      align: "left",
    },
    {
      label: "Cost",
      value: (p: Position) => fmtCurrency(
        parseFloat(p.avg_entry_price) * parseFloat(p.qty),
      ),
      align: "right",
    },
    {
      label: "Current",
      value: (p: Position) => fmtCurrency(
        parseFloat(p.qty) * parseFloat(p.current_price)
      ),
      align: "right",
    },
    {
      label: "% TDY",
      value: (p: Position) => `${fmtPercent(parseFloat(p.change_today))}`,
      align: "right",
      className: (p: Position) => parseFloat(p.change_today) > 0 ? "text-green-400" : "text-red-400"
    },
    {
      label: "TDY $ PL",
      value: (p: Position) => fmtCurrency(
        parseFloat(p.unrealized_intraday_pl),
      ),
      align: "right",
    },
    {
      label: "Net PL %",
      value: (p: Position) => fmtPercent(parseFloat(p.unrealized_plpc)),
      align: "right",
      className: (p: Position) => parseFloat(p.unrealized_plpc) > 0 ? "text-green-400" : "text-red-400"
    },
    {
      label: "Net PL $",
      value: (p: Position) => fmtCurrency(parseFloat(p.unrealized_pl)),
      align: "right",
    }
  ];
  
  return (
    <>
      <VirtualizedTable items={positions} count={count} tableDef={tableDef}/>
    </>
  );
}
