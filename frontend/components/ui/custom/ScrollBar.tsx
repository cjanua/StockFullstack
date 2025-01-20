import {
  Dispatch,
  MouseEventHandler,
  SetStateAction,
  useEffect,
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

  const positionRatio = value / maxValue;

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

  const handleScrollPress: MouseEventHandler = (e) => {
    const clickY = e.clientY;
    const thumbY =
      document
        .getElementsByClassName("scroll-thumb")
        .item(0)
        ?.getBoundingClientRect().top ?? 0;

    if (clickY >= thumbY && clickY <= thumbY + thumbHeight) return;

    const minY = document
      .getElementsByClassName("scroll-area")
      .item(0)
      ?.getBoundingClientRect().top;
    const relativeY = clickY - minY!;

    const ratio = relativeY / areaHeight!;

    setValue(Math.floor(maxValue * ratio));
  };

  const handleScrollGrab: MouseEventHandler = () => {
    setIsGrabbing(true);
  };
  const handleScrollMove = (e: MouseEvent) => {
    if (isGrabbing) {
      const minY = document
        .getElementsByClassName("scroll-area")
        .item(0)
        ?.getBoundingClientRect().top;
      const relativeY = e.clientY - (minY ?? 0);

      const ratio = (relativeY - thumbHeight/2) / (areaHeight - thumbHeight);
      const y = Math.floor(maxValue * ratio);
      let pos = y;
      if (pos < 0) pos = 0;
      if (pos > maxValue) pos = maxValue;
      setValue(pos);
    }
  };
  const handleScrollRelease = (_unused: MouseEvent) => {
    setIsGrabbing(false);
  };

  useEffect(() => {
    if (isEnabled || isGrabbing) {
      const mainScrollBarWidth =
        window.innerWidth - document.documentElement.clientWidth;
      document.body.style.overflowY = "hidden";
      if (mainScrollBarWidth > 0)
        document.body.style.paddingRight = `${mainScrollBarWidth}px`;
    } else {
      document.body.style.overflowY = "scroll";
      document.body.style.paddingRight = "0";
    }
  }, [isEnabled, isGrabbing]);

  document.body.onmouseup = handleScrollRelease;
  document.body.style.userSelect = isGrabbing ? "none" : "auto";
  document.body.onmousemove = handleScrollMove;

  return (
    <div className="w-3" {...props}>
    <div
      className="bg-gray-800 mt-[40px] scroll-area"
      style={{
        height: `${areaHeight}px`,
        width: (isEnabled || isGrabbing) ? "0.75rem" : "0.5rem",
        borderRadius: "0.375rem",
      }}
      onMouseDown={handleScrollPress}
    >
      <div
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
        }}
        onMouseDown={handleScrollGrab}
      />
    </div>
    </div>
  );
}
