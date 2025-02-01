import { execCommand } from "./processes";
import path from "path";
import { existsSync } from "fs";

export async function executeAlpacaCommand<T>(command: string, args: string[] = []): Promise<T> {
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

    const fullArgs = [scriptPath, command, ...args];
    const result = await execCommand<T>(pythonInterpreter, fullArgs);
    console.log(`${command} fetched successfully:`);

    return result;
  } catch (error) {
    console.error(`Error fetching ${command}:`, error);
    throw new Error(
      `${command} fetch error: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
}
