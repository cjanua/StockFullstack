// dashboard/lib/alpaca.ts
"use server";

import { AlpacaClient, formatErrorForLogging } from "@/lib/alpaca-client";
import {
  Account,
  Watchlist,
  PortfolioHistory,
  Order,
  Position,
} from "@/types/alpaca";
import { Clock, StocksQuotesLatest } from "@alpacahq/typescript-sdk";
import Database from "better-sqlite3";
import path from "path";

export async function getAlpacaClientForUser(userId: string): Promise<AlpacaClient> {
  const { alpaca_key, alpaca_secret, use_paper_trading } = await getUserKeys(userId);
  if (!alpaca_key || !alpaca_secret) {
    throw new Error("User Alpaca credentials not configured");
  }
  return new AlpacaClient(alpaca_key, alpaca_secret, use_paper_trading === 1);
}

async function getUserKeys(userId: string): Promise<{
  alpaca_key: string | null;
  alpaca_secret: string | null;
  use_paper_trading: number;
}> {
  const dbPath = path.resolve(process.cwd(), "data/auth.db");
  const db = new Database(dbPath, { readonly: false });
  try {
    const stmt = db.prepare("SELECT alpaca_key, alpaca_secret, use_paper_trading FROM users WHERE id = ?");
    const user = stmt.get(userId) as { alpaca_key: string | null; alpaca_secret: string | null; use_paper_trading: number };
    return user || { alpaca_key: null, alpaca_secret: null, use_paper_trading: 1 };
  } finally {
    db.close();
  }
}

export async function getAlpacaAccount(userId: string): Promise<Account> {
  try {
    const client = await getAlpacaClientForUser(userId);
    return await client.getAccount();
  } catch (error) {
    console.error(`Error fetching account for user ${userId}:`, formatErrorForLogging(error));
    throw error;
  }
}

export async function getAlpacaPortfolioHistory(
  userId: string,
  days: number = 7,
  timeframe: string = "1D"
): Promise<PortfolioHistory> {
  try {
    const client = await getAlpacaClientForUser(userId);
    return await client.getPortfolioHistory({ period: `${days}d`, timeframe });
  } catch (error) {
    console.error(`Error fetching portfolio history for user ${userId}:`, formatErrorForLogging(error));
    throw error;
  }
}

export async function getAlpacaWatchlists(userId: string): Promise<Watchlist[]> {
  try {
    const client = await getAlpacaClientForUser(userId);
    return await client.getWatchlists();
  } catch (error) {
    console.error(`Error fetching watchlists for user ${userId}:`, formatErrorForLogging(error));
    throw error;
  }
}

export async function getAlpacaPositions(userId: string): Promise<Position[]> {
  try {
    const client = await getAlpacaClientForUser(userId);
    return await client.getPositions();
  } catch (error) {
    console.error(`Error fetching positions for user ${userId}:`, formatErrorForLogging(error));
    throw error;
  }
}

export async function closeAlpacaPosition(
  userId: string,
  symbol: string,
  qty: string,
  side: "long" | "short" = "long"
): Promise<Order> {
  try {
    if (!qty) {
      throw new Error(`Quantity is required to close position for ${symbol}`);
    }
    const client = await getAlpacaClientForUser(userId);
    const orderParams = {
      symbol: symbol.toUpperCase(),
      qty,
      side: side === "long" ? "sell" : "buy",
      type: "market",
      time_in_force: "day",
    };
    console.log(`Creating order to close ${symbol} position for user ${userId}:`, orderParams);
    return await client.createOrder(orderParams);
  } catch (error) {
    console.error(`Error closing position for ${symbol} for user ${userId}:`, formatErrorForLogging(error));
    throw error;
  }
}

export async function createAlpacaOrder(userId: string, params: any): Promise<Order> {
  try {
    const client = await getAlpacaClientForUser(userId);
    return await client.createOrder(params);
  } catch (error) {
    console.error(`Error creating order for user ${userId}:`, formatErrorForLogging(error));
    throw error;
  }
}

export async function getAlpacaOrders(userId: string, params: any = {}): Promise<Order[]> {
  try {
    const client = await getAlpacaClientForUser(userId);
    return await client.getOrders(params);
  } catch (error) {
    console.error(`Error getting orders for user ${userId}:`, formatErrorForLogging(error));
    throw error;
  }
}

export async function getAlpacaOrder(userId: string, orderId: string): Promise<Order> {
  try {
    const client = await getAlpacaClientForUser(userId);
    return await client.getOrder(orderId);
  } catch (error) {
    console.error(`Error getting order ${orderId} for user ${userId}:`, formatErrorForLogging(error));
    throw error;
  }
}

export async function cancelAlpacaOrder(userId: string, orderId: string): Promise<Order> {
  try {
    const client = await getAlpacaClientForUser(userId);
    return await client.cancelOrder(orderId);
  } catch (error) {
    console.error(`Error canceling order ${orderId} for user ${userId}:`, formatErrorForLogging(error));
    throw error;
  }
}

export async function cancelAllAlpacaOrders(userId: string): Promise<void> {
  try {
    const openOrders = await getAlpacaOrders(userId, { status: "open" });
    const cancelPromises = openOrders.map((order) => cancelAlpacaOrder(userId, order.id));
    await Promise.all(cancelPromises);
  } catch (error) {
    console.error(`Error canceling all orders for user ${userId}:`, formatErrorForLogging(error));
    throw error;
  }
}

export async function getAlpacaLatestQuote(userId: string, symbol: string): Promise<any> {
  try {
    const { alpaca_key, alpaca_secret } = await getUserKeys(userId);
    if (!alpaca_key || !alpaca_secret) {
      throw new Error("User Alpaca credentials not configured");
    }
    const options = {
      method: "GET",
      headers: {
        accept: "application/json",
        "APCA-API-KEY-ID": alpaca_key,
        "APCA-API-SECRET-KEY": alpaca_secret,
      },
    };
    let attempt = 0;
    const maxRetries = 3;
    while (attempt < maxRetries) {
      const res_raw = await fetch(
        `https://data.alpaca.markets/v2/stocks/quotes/latest?symbols=${symbol}`,
        options
      );
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
        console.warn(
          `Too many requests to API. Retrying Latest Quote for ${symbol} for user ${userId}... (${attempt + 1}/${maxRetries})`
        );
        attempt++;
        await new Promise((resolve) => setTimeout(resolve, 1000 * attempt));
      } else {
        const res = await res_raw.json();
        console.error(`Failed to fetch latest quote for ${symbol} for user ${userId}:`, res);
        throw new Error(`Failed to fetch latest quote for ${symbol}`);
      }
    }
    throw new Error(`Exceeded maximum retries for ${symbol}`);
  } catch (error) {
    console.error(`Error getting latest quote for ${symbol} for user ${userId}:`, formatErrorForLogging(error));
    throw error;
  }
}

export async function getAlpacaLatestQuotes(userId: string, symbols: string[]): Promise<any[]> {
  try {
    const { alpaca_key, alpaca_secret } = await getUserKeys(userId);
    if (!alpaca_key || !alpaca_secret) {
      throw new Error("User Alpaca credentials not configured");
    }
    const options = {
      method: "GET",
      headers: {
        accept: "application/json",
        "APCA-API-KEY-ID": alpaca_key,
        "APCA-API-SECRET-KEY": alpaca_secret,
      },
    };
    let attempt = 0;
    const maxRetries = 3;
    while (attempt < maxRetries) {
      const symbolsParam = symbols.map(encodeURIComponent).join("%2C");
      const res_raw = await fetch(
        `https://data.alpaca.markets/v2/stocks/quotes/latest?symbols=${symbolsParam}`,
        options
      );
      if (res_raw.ok) {
        const res: StocksQuotesLatest = await res_raw.json();
        const quotes = symbols.map((symbol) => {
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
        console.warn(
          `Too many requests to API. Retrying Latest Quotes for user ${userId}... (${attempt + 1}/${maxRetries})`
        );
        attempt++;
        await new Promise((resolve) => setTimeout(resolve, 1000 * attempt));
      } else {
        const res = await res_raw.json();
        console.error(`Failed to fetch latest quotes for user ${userId}:`, res);
        throw new Error(`Failed to fetch latest quotes`);
      }
    }
    throw new Error(`Exceeded maximum retries`);
  } catch (error) {
    console.error(`Error getting latest quotes for user ${userId}:`, formatErrorForLogging(error));
    throw error;
  }
}

export async function getAlpacaClock(userId: string): Promise<Clock> {
  try {
    const client = await getAlpacaClientForUser(userId);
    return await client.getClock();
  } catch (error) {
    console.error(`Error fetching market clock for user ${userId}:`, formatErrorForLogging(error));
    throw error;
  }
}