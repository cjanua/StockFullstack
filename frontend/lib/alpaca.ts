import { Account, createClient } from "@alpacahq/typescript-sdk";

import dotenv from 'dotenv'

dotenv.config()

const client = createClient({
    key: process.env.APCA_KEY,
    secret: process.env.APCA_SECRET,
    paper: false,
})

export async function getAlpacaAccount(): Promise<Account> {
    try {
      const account = await client.getAccount()
      return account
    } catch (error) {
      console.error('Error fetching Alpaca account:', error)
      throw error
    }
  }