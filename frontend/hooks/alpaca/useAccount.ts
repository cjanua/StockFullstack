// hooks/useAccount.ts
"use client"
import { Account } from '@alpacahq/typescript-sdk'
import { useState, useEffect } from 'react'

export function useAccount() {
  const [account, setAccount] = useState<Account | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    async function fetchAccount() {
      try {
        const response = await fetch('/api/account')
        if (!response.ok) throw new Error('Failed to fetch account data')
        const data = await response.json()
        setAccount(data)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An error occurred')
      } finally {
        setIsLoading(false)
      }
    }

    fetchAccount()
  }, [])

  return { account, isLoading, error }
}