// lib/alpaca.ts
"use server";
import { Account } from "@alpacahq/typescript-sdk";

import dotenv from "dotenv";
import { execCommand } from "./processes";

dotenv.config();

async function getAlpacaAccount(): Promise<Account> {
  const cmd = `../backend/alpaca/apca.py`;
  const args = ["trading/account", ];
  const account = execCommand<Account>(cmd, args);

  return account;
}

export { getAlpacaAccount, type Account };
