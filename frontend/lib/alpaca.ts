// lib/alpaca.ts
"use server";

import { 
  createClient, CreateClientOptions, Client,
  Account, Watchlist, PortfolioHistory, Direction,
  CreateOrderOptions,
  Order,
  GetOrderOptions
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

// Orders
export async function createAlpacaOrder(params: any): Promise<Order> {
  try {
    return await client.createOrder(params);
  } catch (error) {
    console.error("Error creating order:", error);
    throw error;
  }
}

export async function getAlpacaOrders(params: any = {}): Promise<Order> {
  try {
    return await client.getOrders(params);
  } catch (error) {
    console.error("Error getting orders:", error);
    throw error;
  }
}

// Get a specific order by ID
export async function getAlpacaOrder(orderId: string): Promise<Order> {
  try {
    return await client.getOrder({
      order_id: orderId,
    });
  } catch (error) {
    console.error(`Error getting order ${orderId}:`, error);
    throw error;
  }
}

export async function cancelAlpacaOrder(orderId: string): Promise<Order> {
  try {
    return await client.cancelOrder({
      order_id: orderId,
    });
  } catch (error) {
    console.error(`Error canceling order ${orderId}:`, error);
    throw error;
  }
}

// // Cancel all open orders by fetching them first and canceling each one
export async function cancelAllAlpacaOrders(): Promise<void> {
  try {
    // Get all open orders
    const openOrders = await getAlpacaOrders({ status: 'open' });
    
    // Cancel each order one by one
    const _ = await cancelAlpacaOrder(openOrders.id);
    
    // Wait for all cancel operations to complete
    // await Promise.all(cancelPromises);
    
    return;
  } catch (error) {
    console.error("Error canceling all orders:", error);
    throw error;
  }
}