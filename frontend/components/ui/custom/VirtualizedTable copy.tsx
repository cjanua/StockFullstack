import React, { ReactNode, useRef, useEffect, useState, JSX } from 'react';

export interface ColDef<T> {
  label: ReactNode;
  value: (item: T) => string | ReactNode;
  align?: 'left' | 'center' | 'right';
  className?: string | ((item: T) => string);
  width?: number;
  render?: (item: T) => ReactNode;
}

interface VirtualizedTableProps<T> {
  items: T[];
  count: number;
  tableDef: ColDef<T>[];
  rowHeight?: number;
  containerHeight?: number;
  onRowClick?: (item: T) => void;
}

export default function VirtualizedTable<T>({
  items,
  count,
  tableDef,
  rowHeight = 48,
  containerHeight = 600,
  onRowClick,
}: VirtualizedTableProps<T>): JSX.Element {
  const containerRef = useRef<HTMLDivElement>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const [containerWidth, setContainerWidth] = useState(0);

  useEffect(() => {
    if (containerRef.current) {
      const resizeObserver = new ResizeObserver((entries) => {
        setContainerWidth(entries[0].contentRect.width);
      });
      resizeObserver.observe(containerRef.current);
      return () => {
        resizeObserver.disconnect();
      };
    }
  }, []);

  // Calculate column widths
  const calculatedWidths = tableDef.map((col, i) => {
    if (col.width) return col.width;
    return 1; // Default flex value
  });
  
  const totalFlex = calculatedWidths.reduce((a, b) => a + b, 0);
  
  // Enhanced safety check for items array
  if (!items || !Array.isArray(items)) {
    console.warn("VirtualizedTable received invalid items array:", items);
    return (
      <div className="border rounded-md p-4">
        <div className="text-center text-muted-foreground">No data available</div>
      </div>
    );
  }
  
  // Safely calculate total and visible rows
  const totalRows = Math.min(items.length, count || items.length);
  const visibleRows = Math.ceil(containerHeight / rowHeight);
  const totalHeight = totalRows * rowHeight;
  
  // Calculate which rows to render
  const startIndex = Math.max(0, Math.floor(scrollTop / rowHeight));
  const endIndex = Math.min(totalRows, startIndex + visibleRows + 1);
  
  // Safe slice of items array
  const visibleItems = items.length > 0 
    ? items.slice(startIndex, endIndex) 
    : [];

  // Render table header
  const renderTableHeader = () => {
    return (
      <div className="flex border-b sticky top-0 z-10 bg-background">
        {tableDef.map((col, i) => {
          const flexValue = (calculatedWidths[i] / totalFlex) * 100;
          const alignClass = col.align === 'right' 
            ? 'justify-end'
            : col.align === 'center' ? 'justify-center' : 'justify-start';
          
          return (
            <div
              key={i}
              style={{ flex: `${flexValue}%` }}
              className={`px-4 py-3 text-sm font-medium text-muted-foreground flex items-center ${alignClass}`}
            >
              {col.label}
            </div>
          );
        })}
      </div>
    );
  };

  // Render table body with improved error handling
  const renderTableBody = () => {
    if (!Array.isArray(items) || items.length === 0) {
      return (
        <div className="py-10 text-center">
          <p className="text-muted-foreground">No items to display</p>
        </div>
      );
    }
    
    return (
      <div 
        style={{ 
          height: totalHeight || rowHeight, // Ensure we have at least one row height
          position: 'relative',
        }}
      >
        {visibleItems.map((item, i) => {
          // Additional safety check for item
          if (!item) {
            return null;
          }
          
          const actualIndex = startIndex + i;
          
          return (
            <div
              key={actualIndex}
              className={`flex border-b hover:bg-muted/50 transition-colors ${onRowClick ? 'cursor-pointer' : ''}`}
              style={{
                position: 'absolute',
                top: actualIndex * rowHeight,
                left: 0,
                right: 0,
                height: rowHeight,
              }}
              onClick={() => onRowClick && item && onRowClick(item)}
            >
              {tableDef.map((col, j) => {
                const flexValue = (calculatedWidths[j] / totalFlex) * 100;
                const alignClass = col.align === 'right' 
                  ? 'justify-end'
                  : col.align === 'center' ? 'justify-center' : 'justify-start';
                
                let className = `px-4 py-2 flex items-center ${alignClass}`;
                if (col.className) {
                  try {
                    const customClass = typeof col.className === 'function'
                      ? col.className(item)
                      : col.className;
                    className += ` ${customClass || ''}`;
                  } catch (err) {
                    console.error("Error calculating className:", err);
                  }
                }
                
                // Safe rendering with error boundaries
                let content;
                try {
                  if (col.render) {
                    content = col.render(item);
                  } else if (col.value) {
                    content = col.value(item);
                  } else {
                    content = "—"; // Default display for empty cells
                  }
                } catch (err) {
                  console.error(`Error rendering column ${j}:`, err);
                  content = <span className="text-destructive">Error</span>;
                }
                
                return (
                  <div
                    key={j}
                    style={{ flex: `${flexValue}%` }}
                    className={className}
                  >
                    {content}
                  </div>
                );
              })}
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <div
      ref={containerRef}
      className="border rounded-md overflow-hidden"
      style={{ height: containerHeight }}
    >
      <div 
        className="overflow-auto h-full"
        onScroll={(e) => setScrollTop(e.currentTarget.scrollTop)}
      >
        {renderTableHeader()}
        {renderTableBody()}
      </div>
    </div>
  );
}