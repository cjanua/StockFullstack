// lib/alpaca.ts
"use server";

import { 
  createClient, CreateClientOptions, Client,
  Account, Watchlist, PortfolioHistory,
  Order, Position
} from "@/types/alpaca";
import { Clock, StocksQuotesLatest } from "@alpacahq/typescript-sdk";

import { env } from "process";

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

// Orders with improved error handling
export async function createAlpacaOrder(params: any): Promise<Order> {
  try {
    // Validate the order parameters
    validateOrderParams(params);
    
    // Create the order
    return await client.createOrder(params);
  } catch (error) {
    console.error("Error creating order:", formatErrorForLogging(error));
    
    // Enhance error with more context
    if (error instanceof Error) {
      // Preserve the original error message but add context
      error.message = `Order creation failed: ${error.message}`;
    }
    
    throw error;
  }
}

// Helper function to validate order parameters
function validateOrderParams(params: any): void {
  // Required fields for all order types
  if (!params.symbol) {
    throw new Error("Symbol is required");
  }
  
  if (!params.side || !['buy', 'sell'].includes(params.side.toLowerCase())) {
    throw new Error("Side must be 'buy' or 'sell'");
  }
  
  if (!params.type || !['market', 'limit', 'stop', 'stop_limit'].includes(params.type.toLowerCase())) {
    throw new Error("Type must be one of: 'market', 'limit', 'stop', 'stop_limit'");
  }
  
  if (!params.time_in_force) {
    throw new Error("Time in force is required");
  }
  
  // Check quantity-related fields
  if (!params.qty && !params.notional) {
    throw new Error("Either qty or notional must be provided");
  }
  
  // Limit orders must have a limit price
  if (params.type === 'limit' && !params.limit_price) {
    throw new Error("Limit orders must specify a limit_price");
  }
  
  // Stop orders must have a stop price
  if ((params.type === 'stop' || params.type === 'stop_limit') && !params.stop_price) {
    throw new Error("Stop orders must specify a stop_price");
  }
  
  // Additional validations could be added here
}

// Format error for logging to avoid [custom formatter threw an exception]
function formatErrorForLogging(error: unknown): string {
  try {
    if (error instanceof Error) {
      return error.message;
    }
    return String(error);
  } catch (_) {
    return "Error details could not be formatted";
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

/**
 * Get latest quote for a symbol
 */
export async function getAlpacaLatestQuote(symbol: string): Promise<any> {
  try {
    const options = {
      method: 'GET',
      headers: {
        accept: 'application/json',
        'APCA-API-KEY-ID': `${env.ALPACA_KEY}`,
        'APCA-API-SECRET-KEY': `${env.ALPACA_SECRET}`
      }
    };

    let attempt = 0;
    const maxRetries = 3;

    while (attempt < maxRetries) {
      const res_raw = await fetch(`https://data.alpaca.markets/v2/stocks/quotes/latest?symbols=${symbol}`, options);

      if (res_raw.ok) {
        const res: StocksQuotesLatest = await res_raw.json();
        const quote = res.quotes[symbol];
        if (!quote) {
          throw new Error(`No quote found for ${symbol}`);
        }

        return {
          symbol,
          price: quote.ap,
          timestamp: new Date(quote.t).toISOString(),
        };
      } else if (res_raw.status === 429) {
        console.warn(`Too many requests to API. Retrying Latest Quote for ${symbol}... (${attempt + 1}/${maxRetries})`);
        attempt++;
        await new Promise(resolve => setTimeout(resolve, 1000 * attempt)); // Exponential backoff
      } else {
        const res = await res_raw.json();
        console.error(`Failed to fetch latest quote for ${symbol}:`, res);
        throw new Error(`Failed to fetch latest quote for ${symbol}`);
      }
    }

    throw new Error(`Exceeded maximum retries for ${symbol}`);
  } catch (error) {
    console.error(`Error getting latest quote for ${symbol}:`, formatErrorForLogging(error));
    throw error;
  }
}
export async function getAlpacaLatestQuotes(symbols: string[]): Promise<any[]> {
  try {
    const options = {
      method: 'GET',
      headers: {
        accept: 'application/json',
        'APCA-API-KEY-ID': `${env.ALPACA_KEY}`,
        'APCA-API-SECRET-KEY': `${env.ALPACA_SECRET}`
      }
    };

    let attempt = 0;
    const maxRetries = 3;

    while (attempt < maxRetries) {
      const symbolsParam = symbols.map(encodeURIComponent).join('%2C');
      const res_raw = await fetch(`https://data.alpaca.markets/v2/stocks/quotes/latest?symbols=${symbolsParam}`, options);

      if (res_raw.ok) {
        const res: StocksQuotesLatest = await res_raw.json();
        const quotes = symbols.map(symbol => {
          const quote = res.quotes[symbol];
          if (!quote) {
            throw new Error(`No quote found for ${symbol}`);
          }
          return {
            symbol,
            price: quote.ap,
            timestamp: new Date(quote.t).toISOString(),
          };
        });

        return quotes;
      } else if (res_raw.status === 429) {
        console.warn(`Too many requests to API. Retrying Latest Quotes... (${attempt + 1}/${maxRetries})`);
        attempt++;
        await new Promise(resolve => setTimeout(resolve, 1000 * attempt)); // Exponential backoff
      } else {
        const res = await res_raw.json();
        console.error(`Failed to fetch latest quotes:`, res);
        throw new Error(`Failed to fetch latest quotes`);
      }
    }

    throw new Error(`Exceeded maximum retries`);
  } catch (error) {
    console.error(`Error getting latest quotes:`, formatErrorForLogging(error));
    throw error;
  }
}

// Add a new function to get the market clock
export async function getAlpacaClock(): Promise<Clock> {
  return client.getClock();
}

// Export the AlpacaClient class so it can be used in routes
export { AlpacaClient };

