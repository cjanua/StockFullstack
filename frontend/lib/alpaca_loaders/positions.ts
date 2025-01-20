import { Direction } from "@alpacahq/typescript-sdk";
import { execCommand } from "../processes";
import path from "path";
import { existsSync } from "fs";

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

export default async function getAlpacaPositions(): Promise<Position[]> {
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

    const positions = await execCommand<Position[]>(pythonInterpreter, [
      scriptPath,
      "trading/account/positions",
    ]);
    console.log("Positions fetched successfully:", positions);

    return positions;
  } catch (error) {
    console.error("Error fetching positions:", error);
    throw new Error(
      `Account fetch error: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
}
