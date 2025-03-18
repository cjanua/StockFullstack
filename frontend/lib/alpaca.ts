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


/**
 * Close a position by creating a market order to sell/buy the position
 * @param symbol Symbol of the position to close
 * @param qty Quantity to close (required)
 * @param side Side of the current position ('long' closes with sell, 'short' closes with buy)
 * @returns The created order object
 */
export async function closeAlpacaPosition(symbol: string, qty: string, side: 'long' | 'short' = 'long'): Promise<Order> {
  try {
    // Ensure we have a quantity (required by Alpaca API)
    if (!qty) {
      throw new Error(`Quantity is required to close position for ${symbol}`);
    }
    
    const orderParams = {
      symbol: symbol.toUpperCase(),
      qty: qty, // Always provide quantity
      side: side === 'long' ? 'sell' : 'buy', // Close long positions by selling, short positions by buying
      type: 'market',
      time_in_force: 'day',
    };
    
    console.log(`Creating order to close ${symbol} position:`, orderParams);
    
    // Create a market order to close the position
    return await createAlpacaOrder(orderParams);
  } catch (error) {
    console.error(`Error closing position for ${symbol}:`, error);
    throw error;
  }
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

export async function getAlpacaOrders(params: any = {}): Promise<Order[]> {
  try {
    const orders = await client.getOrders(params);
    return Array.isArray(orders) ? orders : [orders];
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

// Cancel all open orders by fetching them first and canceling each one
export async function cancelAllAlpacaOrders(): Promise<void> {
  try {
    // Get all open orders
    const openOrders = await getAlpacaOrders({ status: 'open' });

    // Cancel each order one by one
    const cancelPromises = openOrders.map(order => cancelAlpacaOrder(order.id));

    // Wait for all cancel operations to complete
    await Promise.all(cancelPromises);

    return;
  } catch (error) {
    console.error("Error canceling all orders:", error);
    throw error;
  }
}