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
import { ScrollBar } from "@/components/ui/custom/ScrollBar";
import { useMouse } from "react-use";
import { ThHTMLAttributes, useRef, useState, WheelEventHandler } from "react";

export type ColDef<T> = {
  label: string;
  value: (item: T) => string;
  align: string;
  cellStyle?: (item: T) => object;
};

export default function VirtualizedTable<T>({ items, count, tableDef }: { items: T[], count: number, tableDef: ColDef<T>[],  }) {
  const [range, setRange] = useState(0);

  const mouseRef = useRef(null!);
  const { elX, elY, elW, elH } = useMouse(mouseRef);

  const isInside = elX > 0 && elX < elW && elY > 0 && elY < elH;

  const tableHeight = document.querySelector(".tbody")?.clientHeight ?? 0;

  const scrollThumbHeight = (count / items.length) * tableHeight;

  const handleWheel: WheelEventHandler = (e) => {
    if (range >= items.length - count && e.deltaY > 0) return;
    if (range == 0 && e.deltaY < 0) return;

    setRange(range + (e.deltaY > 0 ? 1 : -1));
  };
  const width = document.querySelector('.tbody')?.clientWidth ?? 0 / (tableDef.length-1);
  const cellStyle = { width: `${width}px` };
  const THead = (props: ThHTMLAttributes<HTMLTableCellElement>) => <TableHead style={cellStyle} {...props} />;

    return (
    <div className="pr-[10%] w-[110%] overflow-clip position-table">
      <div className="flex" ref={mouseRef}>
        <div className="flex-1">
          <Table>
            <TableCaption>A list of your portfolio items.</TableCaption>
            <TableHeader className="w-max offset">
              <TableRow
                style={{ backgroundColor: "rgba(120, 120, 120, 0)" }}
              >
                {tableDef.map((d) => (
                  <THead key={d.label} style={{ textAlign: d.align as CanvasTextAlign, width: `${width}px`}}>
                    {d.label}
                  </THead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody
              className="border border-gray-700 border-r-0 rounded-3xl tbody"
              onWheel={handleWheel}
              style={{ maxHeight: `${count * 53}px` }}
            >
              {items.slice(range, range+count).map((item, i) => {
                return <TableRow 
                  key = {i}
                  style={{
                    backgroundColor: i % 2 === 0 ? "rgba(120, 120, 120, 0)" : "rgba(120, 120, 120, 0.1)"
                  }}
                >
                  {tableDef.map((def) => (
                    <TableCell
                      key={def.label}
                      className={def.align === "right" ? "text-right" : ""}
                      style={{ width: `${width}px`, ...(def.cellStyle ? def.cellStyle(item) : {})}}
                    >
                      {def.value(item)}
                    </TableCell>
                  ))}
                </TableRow>
              })}
            </TableBody>
            <TableFooter>
              <TableRow>
                <TableCell colSpan={3}>Total</TableCell>
                <TableCell className="text-right">{items.length}</TableCell>
              </TableRow>
            </TableFooter>
          </Table>
        </div>
        <ScrollBar
          areaHeight={tableHeight}
          thumbHeight={scrollThumbHeight}
          value={range}
          setValue={setRange}
          maxValue={items.length - count}
          isEnabled={isInside}
          onWheel={handleWheel}
        />
      </div>
    </div>
    )
}