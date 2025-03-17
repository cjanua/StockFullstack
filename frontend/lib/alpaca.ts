// lib/alpaca.ts
"use server";

import { 
  createClient, CreateClientOptions, Client,
  Account, Watchlist, PortfolioHistory, Direction
} from "@alpacahq/typescript-sdk";

import { env } from "process";


export type Position = {
  asset_id: string;
  exchange: string;
  asset_class: string;
  symbol: string;
  asset_marginable: boolean;
  qty: string;
  avg_entry_price: string;
  side: Direction;
  market_value: string;
  cost_basis: string;
  unrealized_pl: string;
  unrealized_plpc: string;
  unrealized_intraday_pl: string;
  unrealized_intraday_plpc: string;
  current_price: string;
  lastday_price: string;
  change_today: string;
  qty_available: string;
};

class AlpacaClient {
  private static instance: AlpacaClient = new AlpacaClient();
  private client: Client;

  private constructor() {
    const opts: CreateClientOptions = {
      paper: false,
      baseURL: env.ALPACA_URL,
      key: env.ALPACA_KEY,
      secret: env.ALPACA_SECRET,
    };
    this.client = createClient(opts);
  }

  public static getClient(): Client {
    return AlpacaClient.instance.client;
  }
}

const client = AlpacaClient.getClient();

export async function getAlpacaAccount(): Promise<Account> {
  return client.getAccount();
}

export async function getAlpacaPortfolioHistory(days: number = 7, timeframe: string = "1D"): Promise<PortfolioHistory> {
  return client.getPortfolioHistory({
    period: `${days}d`,
    timeframe: timeframe,
  });
}

export async function getAlpacaWatchlists(): Promise<Watchlist[]> {
  return client.getWatchlists();
}

export async function getAlpacaPositions(): Promise<Position[]> {
  return client.getPositions();
}
