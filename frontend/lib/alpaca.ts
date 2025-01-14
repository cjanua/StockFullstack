// lib/alpaca.ts
"use server";
import { Account } from "@alpacahq/typescript-sdk";

import dotenv from "dotenv";
import { execCommand } from "./processes";
import path from "path";

dotenv.config();

async function getAlpacaAccount(): Promise<Account> {
  const projectRoot = process.env.PROJECT_ROOT;
  if (!projectRoot) {
    throw new Error("PROJECT_ROOT environment variable not set");
  }

  const pythonInterpreter = path.join(projectRoot, 'venv', 'bin', 'python');
  const scriptPath = path.join(projectRoot, 'backend', 'alpaca', 'apca.py');
  

  try {
    // Verify file exists
    const fs = require('fs');
    if (!fs.existsSync(pythonInterpreter)) {
      throw new Error(`Python interpreter not found at: ${pythonInterpreter}`);
    }
    if (!fs.existsSync(scriptPath)) {
      throw new Error(`Script not found at: ${scriptPath}`);
    }

    const account = await execCommand<Account>(
      pythonInterpreter,
      [scriptPath, "trading/account"]
    );

    
    console.log("Account fetched successfully:", account);
    return account;
  } catch (error) {
    console.error("Error fetching account:", error);
    throw new Error(`Account fetch error: ${error instanceof Error ? error.message : String(error)}`);
  }
}

export { getAlpacaAccount, type Account };
