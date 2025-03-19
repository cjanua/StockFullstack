"use client"
import { SymbolOverviewNoSSR } from "@/components/plotting/SymbolOverview";
import { useWatchlists } from "@/hooks/alpaca/useWatchlists";
import { useEffect, useState } from "react";
import { useSearchParam } from "react-use"
import "./styles.css";
import { Asset, Watchlist } from "@/types/alpaca";

export default function Home() {
  const watchlist = useSearchParam("watchlist");

  const { watchlists, isLoading, isError, error } = useWatchlists();

  const [symbols, setSymbols] = useState<string[][]>([[]]);
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
  }, []);

  useEffect(() => {
    if (!isLoading && !isError && watchlists) {
      const selectedWatchlist = watchlists.find((w: Watchlist) => w.id === watchlist);
      if (selectedWatchlist && selectedWatchlist.assets) {
        setSymbols(selectedWatchlist.assets.map((a: Asset) => [a.symbol]));
      }
    }
  }, [isLoading, isError, watchlists, watchlist]);

  if (!isClient) {
    return null;
  }

  return (
    <div className="grid place-items-center pt-8">
      <>{watchlist}</>
      <>{isError && error?.fallback}</>
      <div className="symbol-overview-container">
        <SymbolOverviewNoSSR
          dateFormat="MM/dd/yy"
          colorTheme="dark"
          chartType="candlesticks"
          downColor="#800080"
          borderDownColor="#800080"
          wickDownColor="#800080"
          fontSize="20"
          symbols={symbols}
          autosize={true}
        />
      </div>
    </div>
  );
}