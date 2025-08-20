// dashboard/lib/alpaca-client.ts
import {
  createClient,
  CreateClientOptions,
  Client,
  Account,
  Watchlist,
  PortfolioHistory,
  Order,
  Position,
} from "@/types/alpaca";
import { Clock } from "@alpacahq/typescript-sdk";

export class AlpacaClient {
  private client: Client;

  constructor(key: string, secret: string, paper: boolean) {
    if (!key || !secret) {
      throw new Error("Missing credentials (need key and secret)");
    }
    const opts: CreateClientOptions = {
      key,
      secret,
      paper,
      baseURL: paper
        ? "https://paper-api.alpaca.markets"
        : "https://api.alpaca.markets",
    };
    this.client = createClient(opts);
  }

  async getAccount(): Promise<Account> {
    return this.client.getAccount();
  }

  async getPortfolioHistory(params: {
    period?: string;
    timeframe?: string;
  }): Promise<PortfolioHistory> {
    return this.client.getPortfolioHistory(params);
  }

  async getWatchlists(): Promise<Watchlist[]> {
    return this.client.getWatchlists();
  }

  async getPositions(): Promise<Position[]> {
    return this.client.getPositions();
  }

  async createOrder(params: any): Promise<Order> {
    validateOrderParams(params);
    return this.client.createOrder(params);
  }

  async getOrders(params: any = {}): Promise<Order[]> {
    const orders = await this.client.getOrders(params);
    return Array.isArray(orders) ? orders : [orders];
  }

  async getOrder(orderId: string): Promise<Order> {
    return this.client.getOrder({ order_id: orderId });
  }

  async cancelOrder(orderId: string): Promise<Order> {
    return this.client.cancelOrder({ order_id: orderId });
  }

  async getClock(): Promise<Clock> {
    return this.client.getClock();
  }
}

export function validateOrderParams(params: any): void {
  if (!params.symbol) {
    throw new Error("Symbol is required");
  }
  if (!params.side || !["buy", "sell"].includes(params.side.toLowerCase())) {
    throw new Error("Side must be 'buy' or 'sell'");
  }
  if (
    !params.type ||
    !["market", "limit", "stop", "stop_limit"].includes(params.type.toLowerCase())
  ) {
    throw new Error("Type must be one of: 'market', 'limit', 'stop', 'stop_limit'");
  }
  if (!params.time_in_force) {
    throw new Error("Time in force is required");
  }
  if (!params.qty && !params.notional) {
    throw new Error("Either qty or notional must be provided");
  }
  if (params.type === "limit" && !params.limit_price) {
    throw new Error("Limit orders must specify a limit_price");
  }
  if ((params.type === "stop" || params.type === "stop_limit") && !params.stop_price) {
    throw new Error("Stop orders must specify a stop_price");
  }
}

export function formatErrorForLogging(error: unknown): string {
  try {
    if (error instanceof Error) {
      return error.message;
    }
    return String(error);
  } catch (_) {
    return "Error details could not be formatted";
  }
}