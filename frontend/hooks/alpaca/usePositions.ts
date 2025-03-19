"use client";
import { getError } from "@/types/error";
import { Position } from "@/types/alpaca";
import { useState, useEffect } from "react";

export function usePositions() {
  const [positions, setPositions] = useState<Position[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isError, setIsError] = useState<boolean>(false);
  const [error, setError] = useState(getError());

  // Function to fetch positions data
  const fetchPositions = async () => {
    setIsLoading(true);
    try {
      const cachedPositions = localStorage.getItem("positions");
      if (cachedPositions) {
        setPositions(JSON.parse(cachedPositions));
        setIsLoading(false);
      }

      const response = await fetch("/api/alpaca/positions", {
        next: {
          revalidate: 30, // Revalidate every 30 seconds
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
      localStorage.setItem("positions", JSON.stringify(data));
      setIsError(false);
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
  };
  
  // Add the mutate function to manually trigger a refresh
  const mutate = () => {
    fetchPositions();
  };

  useEffect(() => {
    fetchPositions();
    
    // Optional: Set up polling to keep data fresh
    const intervalId = setInterval(() => {
      fetchPositions();
    }, 60000); // Refresh every minute
    
    return () => {
      clearInterval(intervalId);
    };
  }, []);

  return { positions, isLoading, isError, error, mutate };
}