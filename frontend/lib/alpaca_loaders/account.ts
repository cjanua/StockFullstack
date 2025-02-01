import { Account, Watchlist, PortfolioHistory } from "@alpacahq/typescript-sdk";
import AlpacaClient from "./client";

export async function getAlpacaAccount(): Promise<Account> {
  const client = AlpacaClient.getClient();
  return client.getAccount();
}

export async function getAlpacaPortfolioHistory(days: number = 7, timeframe: string = "1D"): Promise<PortfolioHistory> {
  const client = AlpacaClient.getClient();
  return client.getPortfolioHistory({
    period: `${days}d`,
    timeframe: timeframe,
  });
}

export async function getAlpacaWatchlists(): Promise<Watchlist[]> {
  const client = AlpacaClient.getClient();
  return client.getWatchlists();
}