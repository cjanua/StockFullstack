// hooks/usePositions.ts
"use client";
import { getError } from "@/types/error";
import { Position } from "@/lib/alpaca";
import { useState, useEffect } from "react";

export function usePositions() {
  const [positions, setPositions] = useState<Position[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isError, setIsError] = useState<boolean>(false);
  const [error, setError] = useState(getError());

  useEffect(() => {
    async function fetchPositions() {
      try {
        const response = await fetch("/api/alpaca/positions", {
          next: {
            revalidate: 86400,
            tags: ['positions'],
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

    fetchPositions();
  }, []);

  return { positions, isLoading, isError, error };
}
