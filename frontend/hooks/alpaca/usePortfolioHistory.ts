"use client";
import { getError } from "@/types/error";
import { PortfolioHistory } from "@/types/alpaca";
import { useState, useEffect } from "react";

export function usePortfolioHistory(days: number, timeframe: string) {
  const [portfolioHistory, setPortfolioHistory] = useState<PortfolioHistory | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isError, setIsError] = useState<boolean>(false);
  const [error, setError] = useState(getError());

  useEffect(() => {
    async function fetchPortfolioHistory() {
      try {
        const cacheKey = `portfolioHistory_${days}_${timeframe}`;
        const cachedPortfolioHistory = localStorage.getItem(cacheKey);
        if (cachedPortfolioHistory) {
          setPortfolioHistory(JSON.parse(cachedPortfolioHistory));
          setIsLoading(false);
        }

        const response = await fetch("/api/alpaca/account/history", { 
          headers: {
            "timeframe": `${timeframe}`,
            "days": `${days}`,
          },
          next: {
            revalidate: 86400,
            tags: ['accountHistory'],
          }
        });

        if (!response.ok) {
          setIsError(true);
          setError(getError("nextApiError", response.statusText));
          return;
        }
        const data = await response.json();
        setPortfolioHistory(data);
        localStorage.setItem(cacheKey, JSON.stringify(data));
      } catch (err) {
        setIsError(true);
        if (err instanceof Error) {
          setError(getError("unknownError", err.message));
        } else {
          setError(getError("unknownError", String(err)));
        }
      } finally {
        setIsLoading(false);
      }
    }

    fetchPortfolioHistory();
  }, [days, timeframe]);

  return { portfolioHistory, isLoading, isError, error };
}
