"use client"
import { SymbolOverviewNoSSR } from "@/components/plotting/SymbolOverview";

export default function Home() {
  const viewHeight = document.body.clientHeight * 2 / 3;
  const viewWidth = document.body.clientWidth * 0.75;
  return (
    <div className="grid place-items-center pt-8">
      <SymbolOverviewNoSSR
        dateFormat="MM/dd/yy"
        colorTheme="dark"
        height={viewHeight}
        width={viewWidth}
        chartType="candlesticks"
        downColor="#800080"
        borderDownColor="#800080"
        wickDownColor="#800080"
      />
    </div>
  );
}
