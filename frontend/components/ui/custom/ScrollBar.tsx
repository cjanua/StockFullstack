import {
  Dispatch,
  MouseEventHandler,
  SetStateAction,
  useEffect,
  useRef,
  useState,
} from "react";
import { useKey } from "react-use";

export function ScrollBar({
  areaHeight,
  thumbHeight,
  value,
  setValue,
  maxValue,
  isEnabled,
  ...props
}: {
  areaHeight: number;
  thumbHeight: number;
  value: number;
  setValue: Dispatch<SetStateAction<number>>;
  maxValue: number;
  isEnabled: boolean;
  [key: string]: unknown;
}) {
  const [isGrabbing, setIsGrabbing] = useState(false);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const scrollThumbRef = useRef<HTMLDivElement>(null);
  const dragStartRef = useRef<{y: number, startValue: number} | null>(null);
  
  // Safely calculate position ratio
  const positionRatio = Math.min(1, Math.max(0, value / Math.max(1, maxValue)));

  // Handle keyboard navigation
  useKey(
    "ArrowUp",
    () => {
      if (value > 0 && isEnabled) {
        setValue(value - 1);
      }
    },
    {},
    [value, isEnabled],
  );

  useKey(
    "ArrowDown",
    () => {
      if (value < maxValue && isEnabled) {
        setValue(value + 1);
      }
    },
    {},
    [value, isEnabled],
  );

  // Safe handling of scrollbar track clicks
  const handleScrollPress: MouseEventHandler = (e) => {
    try {
      if (!scrollThumbRef.current || !scrollAreaRef.current) return;
      
      const clickY = e.clientY;
      const thumbRect = scrollThumbRef.current.getBoundingClientRect();
      
      // If the click is on the thumb, don't do anything (drag will handle it)
      if (clickY >= thumbRect.top && clickY <= thumbRect.bottom) return;
      
      const areaRect = scrollAreaRef.current.getBoundingClientRect();
      const relativeY = clickY - areaRect.top;
      
      // Calculate the new position
      let ratio = relativeY / areaHeight;
      ratio = Math.min(1, Math.max(0, ratio)); // Clamp between 0 and 1
      
      setValue(Math.round(maxValue * ratio));
    } catch (err) {
      console.error("ScrollBar click error:", err);
      // Don't update on error
    }
  };

  // Handle thumb dragging - start
  const handleScrollGrab: MouseEventHandler = (e) => {
    e.preventDefault();
    e.stopPropagation();
    dragStartRef.current = {
      y: e.clientY,
      startValue: value
    };
    setIsGrabbing(true);
  };
  
  // Handle mouse movement during dragging
  const handleScrollMove = (e: MouseEvent) => {
    try {
      if (!isGrabbing || !scrollAreaRef.current || !dragStartRef.current) return;
      
      // Calculate the movement delta from the start of the drag
      const deltaY = e.clientY - dragStartRef.current.y;
      
      // Calculate the corresponding value change based on the ratio
      const valueRange = maxValue;
      const pixelRange = areaHeight - thumbHeight;
      const valueChange = Math.round((deltaY / pixelRange) * valueRange);
      
      // Calculate the new value
      let newValue = dragStartRef.current.startValue + valueChange;
      
      // Clamp the value
      newValue = Math.min(maxValue, Math.max(0, newValue));
      
      setValue(newValue);
    } catch (err) {
      console.error("ScrollBar drag error:", err);
    }
  };

  // Handle drag end
  const handleScrollRelease = () => {
    setIsGrabbing(false);
    dragStartRef.current = null;
  };

  // Set up and clean up event listeners
  useEffect(() => {
    const handleGlobalMouseUp = () => {
      if (isGrabbing) {
        setIsGrabbing(false);
        dragStartRef.current = null;
      }
    };
    
    const handleGlobalMouseMove = (e: MouseEvent) => {
      if (isGrabbing) {
        handleScrollMove(e);
      }
    };
    
    document.addEventListener('mouseup', handleGlobalMouseUp);
    document.addEventListener('mousemove', handleGlobalMouseMove);
    
    return () => {
      document.removeEventListener('mouseup', handleGlobalMouseUp);
      document.removeEventListener('mousemove', handleGlobalMouseMove);
    };
  }, [isGrabbing]);

  // Handle document body styles
  useEffect(() => {
    if (isEnabled || isGrabbing) {
      // Calculate scrollbar width to prevent layout shift
      const mainScrollBarWidth = 
        window.innerWidth - document.documentElement.clientWidth;
      
      // Disable scrolling while using custom scrollbar
      document.body.style.overflowY = "hidden";
      
      // Handle cursor and user selection during grabbing
      if (isGrabbing) {
        document.body.style.cursor = "grabbing";
        document.body.style.userSelect = "none";
      } else {
        document.body.style.cursor = "";
        document.body.style.userSelect = "";
      }
      
      // Add padding to prevent layout shift when scrollbar disappears
      if (mainScrollBarWidth > 0) {
        document.body.style.paddingRight = `${mainScrollBarWidth}px`;
      }
    } else {
      // Restore normal scrolling
      document.body.style.overflowY = "scroll";
      document.body.style.paddingRight = "0";
    }
    
    // Clean up body styles on unmount
    return () => {
      document.body.style.overflowY = "";
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      document.body.style.paddingRight = "";
    };
  }, [isEnabled, isGrabbing]);

  return (
    <div className="w-3" {...props}>
      <div
        ref={scrollAreaRef}
        className="bg-gray-800 mt-[40px] scroll-area"
        style={{
          height: `${areaHeight}px`,
          width: (isEnabled || isGrabbing) ? "0.75rem" : "0.5rem",
          borderRadius: "0.375rem",
        }}
        onMouseDown={handleScrollPress}
      >
        <div
          ref={scrollThumbRef}
          className="scroll-thumb"
          style={{
            height: `${thumbHeight}px`,
            transform: `translateY(${positionRatio * (areaHeight - thumbHeight)}px)`,
            background: isGrabbing
              ? "#cccccc"
              : isEnabled
                ? "#999999"
                : "#666666",
            borderRadius: "0.375rem",
            cursor: isGrabbing ? "grabbing" : "grab",
          }}
          onMouseDown={handleScrollGrab}
        />
      </div>
    </div>
  );
}