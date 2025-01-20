import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableFooter,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { usePositions } from "@/hooks/alpaca/usePositions";
import { ScrollBar } from "@/components/ScrollBar";
import { fmtCurrency, fmtPercent } from "@/lib/utils";
import { useMouse } from "react-use";
import { ThHTMLAttributes, useRef, useState, WheelEventHandler } from "react";

export function PositionTable({ count }: { count: number }) {
  const { positions, isLoading, isError, error } = usePositions();

  const [range, setRange] = useState(0);

  const mouseRef = useRef(null!);
  const { elX, elY, elW, elH } = useMouse(mouseRef);

  const isInside = elX > 0 && elX < elW && elY > 0 && elY < elH;

  const tableHeight = document.querySelector(".tbody")?.clientHeight ?? 0;
  const scrollThumbHeight = (count / positions.length) * tableHeight;

  const handleWheel: WheelEventHandler = (e) => {
    if (range >= positions.length - count && e.deltaY > 0) return;
    if (range == 0 && e.deltaY < 0) return;

    setRange(range + (e.deltaY > 0 ? 1 : -1));
  };

  if (isLoading) return <div>Loading positions data...</div>;
  if (isError) return error.fallback;
  if (!positions) return <div>No positions data available</div>;

  const width = document.querySelector('.tbody')?.clientWidth ?? 0 / 6;
  const cellStyle = { width: `${width}px` };
  const THead = (props: ThHTMLAttributes<HTMLTableCellElement>) => <TableHead style={cellStyle} {...props} />;

  return (
    <div className="pr-[10%] w-[110%] overflow-clip position-table">
      <div className="flex" ref={mouseRef}>
        <div className="flex-1">
          <Table>
            <TableCaption>A list of your portfolio positions.</TableCaption>
            <TableHeader className="w-max offset">
              <TableRow
                className="w-max"
                style={{ backgroundColor: "rgba(120, 120, 120, 0)" }}
              >
                <THead >Symbol</THead>
                <THead className="text-right">Cost</THead>
                <THead className="text-right">Current</THead>
                <THead className="text-right">% TDY</THead>
                <THead className="text-right">TDY $ PL</THead>
                <THead className="text-right">Net PL %</THead>
                <THead className="text-right">Net PL $</THead>
              </TableRow>
            </TableHeader>
            <TableBody
              className="border border-gray-700 border-r-0 rounded-3xl tbody"
              onWheel={handleWheel}
              style={{ maxHeight: `${count * 53}px` }}
            >
              {positions.map((p, i) => {
                if (i < range || i >= range + count) return null;
                const value = parseFloat(p.qty) * parseFloat(p.current_price);
                const style =
                  i % 2 === 0
                    ? {backgroundColor: "rgba(120, 120, 120, 0)"}
                    : { backgroundColor: "rgba(120, 120, 120, 0.1)" };

                return (
                  <TableRow key={p.asset_id} style={style}>
                    <TableCell className="font-medium">{p.symbol}</TableCell>
                    <TableCell className="text-right">
                      {fmtCurrency(
                        parseFloat(p.avg_entry_price) * parseFloat(p.qty),
                      )}
                    </TableCell>
                    <TableCell className="text-right">
                      {fmtCurrency(value)}
                    </TableCell>
                    <TableCell className="text-right">{`${fmtPercent(parseFloat(p.change_today))}`}</TableCell>
                    <TableCell className="text-right">
                      {fmtCurrency(
                        parseFloat(p.lastday_price) -
                          parseFloat(p.current_price),
                      )}
                    </TableCell>
                    <TableCell
                      className="text-right"
                      style={{
                        color:
                          parseFloat(p.unrealized_pl) >= 0 ? "green" : "red",
                      }}
                    >
                      {fmtPercent(parseFloat(p.unrealized_plpc))}
                    </TableCell>
                    <TableCell className="text-right">
                      {fmtCurrency(parseFloat(p.unrealized_pl))}
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
            <TableFooter>
              <TableRow>
                <TableCell colSpan={3}>Total</TableCell>
                <TableCell className="text-right">{positions.length}</TableCell>
              </TableRow>
            </TableFooter>
          </Table>
        </div>
        <ScrollBar
          areaHeight={tableHeight}
          thumbHeight={scrollThumbHeight}
          value={range}
          setValue={setRange}
          maxValue={positions.length - count}
          isEnabled={isInside}
          onWheel={handleWheel}
        />
      </div>
    </div>
  );
}
