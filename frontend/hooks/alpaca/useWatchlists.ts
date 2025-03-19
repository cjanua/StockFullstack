"use client";
import { getError } from "@/types/error";
import { Watchlist } from "@/types/alpaca";
import { useState, useEffect } from "react";

export function useWatchlists() {
  const [watchlists, setWatchlists] = useState<Watchlist[] | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isError, setIsError] = useState<boolean>(false);
  const [error, setError] = useState(getError());

  useEffect(() => {
    async function fetchWatchlists() {
      try {
        const cachedWatchlists = localStorage.getItem("watchlists");
        if (cachedWatchlists) {
          setWatchlists(JSON.parse(cachedWatchlists));
          setIsLoading(false);
        }

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
        setWatchlists(data);
        localStorage.setItem("watchlists", JSON.stringify(data));
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

    fetchWatchlists();
  }, []);

  return { watchlists, isLoading, isError, error };
}
