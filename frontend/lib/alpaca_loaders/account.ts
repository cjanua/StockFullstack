import { execCommand } from "../processes";
import path from "path";
import { Account } from "@alpacahq/typescript-sdk";
import { existsSync } from "fs";

export async function getAlpacaAccount(): Promise<Account> {
  const projectRoot = process.env.PROJECT_ROOT;
  if (!projectRoot)
    throw new Error("PROJECT_ROOT environment variable not set");

  const pythonInterpreter = path.join(projectRoot, "venv", "bin", "python");
  const scriptPath = path.join(projectRoot, "backend", "alpaca", "apca.py");

  try {
    // Verify file exists
    if (!existsSync(pythonInterpreter))
      throw new Error(`Python interpreter not found at: ${pythonInterpreter}`);
    if (!existsSync(scriptPath))
      throw new Error(`Script not found at: ${scriptPath}`);

    const account = await execCommand<Account>(pythonInterpreter, [
      scriptPath,
      "trading/account",
    ]);
    console.log("Account fetched successfully:", account);

    return account;
  } catch (error) {
    console.error("Error fetching account:", error);
    throw new Error(
      `Account fetch error: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
}

export async function getAlpacaAccountHistory(days: number = 7, timeframe: string = "1D"): Promise<Account> {
  const projectRoot = process.env.PROJECT_ROOT;
  if (!projectRoot)
    throw new Error("PROJECT_ROOT environment variable not set");

  const pythonInterpreter = path.join(projectRoot, "venv", "bin", "python3");
  const scriptPath = path.join(projectRoot, "backend", "alpaca", "apca.py");

  try {
    // Verify file exists
    if (!existsSync(pythonInterpreter))
      throw new Error(`Python interpreter not found at: ${pythonInterpreter}`);
    if (!existsSync(scriptPath))
      throw new Error(`Script not found at: ${scriptPath}`);
    
    const account = await execCommand<Account>(pythonInterpreter, [
      scriptPath,
      `trading/account/history`,
      `--days`, `${(days)}`,
      `--timeframe`, `${(timeframe)}`,
    ]);
    console.log("Account fetched successfully:", account);

    return account;
  } catch (error) {
    console.error("Error fetching account:", error);
    throw new Error(
      `Account fetch error: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
}
