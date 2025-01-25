// hooks/useAccount.ts
"use client";
import { getError } from "@/types/error";
import { Account } from "@alpacahq/typescript-sdk";
import { useState, useEffect } from "react";

export function useAccount() {
  const [account, setAccount] = useState<Account | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isError, setIsError] = useState<boolean>(false);
  const [error, setError] = useState(getError());

  useEffect(() => {
    async function fetchAccount() {
      try {
        const response = await fetch("/api/alpaca/account", {
          next: {
            revalidate: 86400,
            tags: ['account'],
          }
        });
        if (!response.ok) {
          setIsError(true);
          setError(getError("nextApiError", response.statusText));
          return;
        }
        const data = await response.json();
        setAccount(data);
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

    fetchAccount();
  }, []);

  return { account, isLoading, isError, error };
}
