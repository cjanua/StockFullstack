"use client"
import { SymbolOverviewNoSSR } from "@/components/plotting/SymbolOverview";
import { useWatchlists } from "@/hooks/alpaca/useWatchlists";
import { Asset, Watchlist } from "@alpacahq/typescript-sdk";
import { useEffect, useState } from "react";
import { useWindowSize, useSearchParam } from "react-use"

export default function Home() {
  const watchlist = useSearchParam("watchlist");

  const { width, height } = useWindowSize();

  const { watchlists, isLoading, isError, error } = useWatchlists();

  const [symbols, setSymbols] = useState<string[][]>([[]]);

  useEffect(() => {
    if (!isLoading && !isError && watchlists) {
      const selectedWatchlist = watchlists.find((w: Watchlist) => w.id === watchlist);
      if (selectedWatchlist && selectedWatchlist.assets) {
        setSymbols(selectedWatchlist.assets.map((a: Asset) => [a.symbol]));
      }
    }
  }, [isLoading, isError, watchlists, watchlist]);
  
  return (
    <div className="grid place-items-center pt-8">
      <>{watchlist}</>
      <>{isError && error?.fallback}</>
      <SymbolOverviewNoSSR
        dateFormat="MM/dd/yy"
        colorTheme="dark"
        height={Math.floor(height * 0.75)}
        width={Math.floor(width*0.8)}
        chartType="candlesticks"
        downColor="#800080"
        borderDownColor="#800080"
        wickDownColor="#800080"
        fontSize="20"
        symbols={ symbols }
      />
    </div>
  );
}
