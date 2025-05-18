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
import { ReactNode, ThHTMLAttributes, useEffect, useRef, useState, WheelEventHandler } from "react";

export type ColDef<T> = {
  label: string | ReactNode;
  value: (item: T) => string;
  align: string;
  className?: (item: T) => string;
  render?: (item: T) => ReactNode; // Support for custom rendering
};

export default function VirtualizedTable<T>({ 
  items, 
  count, 
  tableDef,
  footerContent
}: { 
  items: T[], 
  count: number, 
  tableDef: ColDef<T>[],
  footerContent?: ReactNode  
}) {
  const [range, setRange] = useState(0);
  const [errorState, setErrorState] = useState(false);
  const [lastGoodItems, setLastGoodItems] = useState<T[]>([]);
  
  const mouseRef = useRef(null!);
  const { elX, elY, elW, elH } = useMouse(mouseRef);
  const isInside = elX > 0 && elX < elW && elY > 0 && elY < elH;
  
  // Safely get table height and handle potential DOM issues
  const getTableHeight = () => {
    try {
      return document.querySelector(".tbody")?.clientHeight ?? 300; // Fallback height
    } catch (e) {
      console.error("Error getting table height:", e);
      return 300; // Fallback height
    }
  };

  const tableHeight = getTableHeight();
  
  // Safely calculate scroll thumb height with fallbacks
  const getScrollThumbHeight = () => {
    const itemCount = Math.max(items.length, 1);
    // Ensure thumb is at least 30px high for usability
    return Math.max(30, (count / itemCount) * tableHeight);
  };

  const scrollThumbHeight = getScrollThumbHeight();

  // Store items safely when they change
  useEffect(() => {
    if (items && items.length > 0) {
      setLastGoodItems(items);
      if (errorState) setErrorState(false);
    }
  }, [items, errorState]);

  // Ensure range is valid when items change
  useEffect(() => {
    if (items && items.length > 0) {
      const maxValidRange = Math.max(0, items.length - count);
      if (range > maxValidRange) {
        setRange(maxValidRange);
      }
    }
  }, [items, count, range]);

  // Safe wheel event handler with error protection
  const handleWheel: WheelEventHandler = (e) => {
    try {
      // Only handle vertical scrolling, ignore horizontal
      if (Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
        return; // Ignore predominately horizontal scrolling
      }
      
      e.preventDefault(); // Prevent browser scroll when handling vertical
      
      const maxRange = Math.max(0, items.length - count);
      let newRange = range;

      if (e.deltaY > 0) {
        // Scrolling down
        newRange = Math.min(maxRange, range + 1);
      } else if (e.deltaY < 0) {
        // Scrolling up
        newRange = Math.max(0, range - 1);
      }

      setRange(newRange);
    } catch (err) {
      console.error("Scroll handler error:", err);
      // Don't update state on error
    }
  };

  // Safe rendering with error recovery
  const renderTableBody = () => {
    try {
      // Determine which items to display
      const displayItems = items && items.length > 0 
        ? items 
        : (lastGoodItems.length > 0 ? lastGoodItems : []);
        
      if (displayItems.length === 0) {
        return (
          <TableRow>
            <TableCell colSpan={tableDef.length} className="text-center py-8">
              No items to display
            </TableCell>
          </TableRow>
        );
      }
      
      // Ensure range is valid
      const safeRange = Math.min(Math.max(0, range), Math.max(0, displayItems.length - count));
      const visibleItems = displayItems.slice(safeRange, safeRange + count);
      
      // Always show at least one item (prevents empty table)
      if (visibleItems.length === 0 && displayItems.length > 0) {
        visibleItems.push(displayItems[0]);
      }
      
      return visibleItems.map((item, i) => (
        <TableRow 
          key={i}
          style={{
            backgroundColor: i % 2 === 0 ? "rgba(120, 120, 120, 0)" : "rgba(120, 120, 120, 0.1)"
          }}
        >
          {tableDef.map((def, j) => (
            <TableCell
              key={j}
              className={`subpixel-antialiased font-medium ${def.align === "right" ? "text-right" : ""} ${def.className ? def.className(item) : ""}`}
              style={{ width: `${width}px`}}
            >
              {def.render ? def.render(item) : def.value(item)}
            </TableCell>
          ))}
        </TableRow>
      ));
    } catch (err) {
      console.error("Table rendering error:", err);
      setErrorState(true);
      
      // Fallback rendering
      return (
        <TableRow>
          <TableCell colSpan={tableDef.length} className="text-center py-8">
            Error displaying items. Try refreshing the page.
          </TableCell>
        </TableRow>
      );
    }
  };

  // Get the table width
  const width = (() => {
    try {
      return document.querySelector('.tbody')?.clientWidth ?? 0 / (tableDef.length-1);
    } catch (e) {
      console.error("Error getting table width:", e);
      return 100; // Fallback width
    }
  })();

  const THead = (props: ThHTMLAttributes<HTMLTableCellElement>) => <TableHead style={cellStyle} {...props} />;
  const cellStyle = { width: `${width}px` };

  return (
    <div className="pr-[10%] w-[110%] overflow-hidden position-table">
      <div className="flex" ref={mouseRef}>
        <div className="flex-1">
          <Table>
            <TableCaption>A list of your portfolio items.</TableCaption>
            <TableHeader className="w-max offset">
              <TableRow
                style={{ backgroundColor: "rgba(120, 120, 120, 0)" }}
              >
                {tableDef.map((d, i) => (
                  <THead key={i} style={{ textAlign: d.align as CanvasTextAlign, width: `${width}px`}}>
                    {d.label}
                  </THead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody
              className="border border-gray-700 border-r-0 rounded-3xl tbody"
              onWheel={handleWheel}
              style={{ maxHeight: `${count * 53}px`, minHeight: "106px" }} // Ensure at least 2 rows height
            >
              {renderTableBody()}
            </TableBody>
            <TableFooter>
              <TableRow>
                {footerContent ? (
                  footerContent
                ) : (
                  <>
                    <TableCell colSpan={3}>Total</TableCell>
                    <TableCell className="text-right">{items?.length || 0}</TableCell>
                  </>
                )}
              </TableRow>
            </TableFooter>
          </Table>
        </div>
        <ScrollBar
          areaHeight={tableHeight}
          thumbHeight={scrollThumbHeight}
          value={range}
          setValue={(newRange) => {
            // Ensure range is always valid
            const maxValidRange = Math.max(0, (items?.length || 0) - count);
            const safeRange = Math.min(Math.max(0, typeof newRange === 'function' ? newRange(range) : newRange), maxValidRange);
            setRange(safeRange);
          }}
          maxValue={Math.max(0, (items?.length || 0) - count)}
          isEnabled={isInside && items && items.length > count}
          onWheel={handleWheel}
        />
      </div>
    </div>
  );
}