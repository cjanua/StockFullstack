import { executeAlpacaCommand } from "./commandExecutor";
import { Account, Watchlist, PortfolioHistory } from "@alpacahq/typescript-sdk";

export async function getAlpacaAccount(): Promise<Account> {
  return executeAlpacaCommand<Account>("trading/account");
}

export async function getAlpacaAccountHistory(days: number = 7, timeframe: string = "1D"): Promise<PortfolioHistory> {
  return executeAlpacaCommand<PortfolioHistory>("trading/account/history", [
    "--days", `${days}`,
    "--timeframe", `${timeframe}`,
  ]);
}

export async function getAlpacaWatchlists(): Promise<Watchlist[]> {
  return executeAlpacaCommand<Watchlist[]>("trading/account/watchlists");
}