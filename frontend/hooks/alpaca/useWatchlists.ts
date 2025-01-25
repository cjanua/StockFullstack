// hooks/usePositions.ts
"use client";
import { getError } from "@/types/error";
import { Watchlist } from "@alpacahq/typescript-sdk";
import { useState, useEffect } from "react";

export function useWatchlists() {
  const [watchlists, setPositions] = useState<Watchlist[] | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isError, setIsError] = useState<boolean>(false);
  const [error, setError] = useState(getError());

  useEffect(() => {
    async function fetchWatchlists() {
      try {
        const response = await fetch("/api/alpaca/account/watchlists", {
          next: {
            revalidate: 86400,
            tags: ['watchlists'],
          }
        });

        if (!response.ok) {
          setIsError(true);
          setError(getError("nextApiError", response.statusText));
          return;
        }
        const data: Watchlist[] = await response.json();
        console.log(data)
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

    fetchWatchlists();
  }, []);

  return { watchlists, isLoading, isError, error };
}
