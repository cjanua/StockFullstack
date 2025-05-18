import { Account, Asset, Watchlist, CreateOrderOptions, PortfolioHistory,
  createClient, CreateClientOptions, Client, Direction,
  Order,
  GetOrderOptions
 } from "@alpacahq/typescript-sdk";

export type { Account, Asset, Watchlist, PortfolioHistory, Client, Direction, Order, GetOrderOptions };

export type { CreateOrderOptions, CreateClientOptions };

export { createClient };

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