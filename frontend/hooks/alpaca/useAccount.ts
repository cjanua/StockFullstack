// hooks/useAccount.ts
"use client";
import { getError } from "@/types/error";
import { Account } from "@/types/alpaca";
import { useState, useEffect } from "react";

export function useAccount() {
  const [account, setAccount] = useState<Account | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isError, setIsError] = useState<boolean>(false);
  const [error, setError] = useState(getError());

  useEffect(() => {
    async function fetchAccount() {
      try {
        const cachedAccount = localStorage.getItem("account");
        if (cachedAccount) {
          setAccount(JSON.parse(cachedAccount));
          setIsLoading(false);
        }

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
        localStorage.setItem("account", JSON.stringify(data));
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

    fetchAccount();
  }, []);

  return { account, isLoading, isError, error };
}
