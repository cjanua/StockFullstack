"use client"
import { SymbolOverviewNoSSR } from "@/components/plotting/SymbolOverview";
import { useEffect, useState } from "react";

export default function Home() {
  const [ view, setView ] = useState({ height: document.body.clientHeight, width: document.body.clientWidth * 0.75 });

  useEffect(() => {
    const handleResize = () => {
      setView({ height: document.body.clientHeight, width: document.body.clientWidth * 0.75 });
    };
    window.addEventListener("resize", handleResize);
    return () => {
      window.removeEventListener("resize", handleResize);
    };
  }, []);
  return (
    <div className="grid place-items-center pt-8">
      <SymbolOverviewNoSSR
        dateFormat="MM/dd/yy"
        colorTheme="dark"
        height={view.height}
        width={view.width}
        chartType="candlesticks"
        downColor="#800080"
        borderDownColor="#800080"
        wickDownColor="#800080"
      />
    </div>
  );
}
