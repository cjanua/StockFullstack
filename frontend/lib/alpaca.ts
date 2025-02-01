// lib/alpaca.ts
"use server";
import { getAlpacaAccount, getAlpacaAccountHistory, getAlpacaWatchlists } from "./alpaca_loaders/account";
import getAlpacaPositions, { Position } from "./alpaca_loaders/positions";

import { Account, PortfolioHistory, Watchlist } from "@alpacahq/typescript-sdk";

export { getAlpacaAccount, type Account };
export { getAlpacaPositions, type Position };
export { getAlpacaAccountHistory, type PortfolioHistory };
export { getAlpacaWatchlists, type Watchlist };
