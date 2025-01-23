// hooks/usePositions.ts
"use client";
import { getError } from "@/types/error";
import { PortfolioHistory } from "@alpacahq/typescript-sdk";
import { useState, useEffect } from "react";

export function usePortfolioHistory(days: number, timeframe: string) {
  const [portfolioHistory, setPositions] = useState<PortfolioHistory | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isError, setIsError] = useState<boolean>(false);
  const [error, setError] = useState(getError());

  useEffect(() => {
    async function fetchPortfolioHistory() {
      try {
        const response = await fetch("/api/account/history", { 
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
        setPositions(data);
      } catch (err) {
        setIsError(true);
        if (err instanceof Error) {
          setError(getError("unknownError", err.message));
        }
        setError(getError("unknownError", String(err)));
      } finally {
        setIsLoading(false);
      }
    }

    fetchPortfolioHistory();
  }, [days, timeframe]);

  return { portfolioHistory, isLoading, isError, error };
}
