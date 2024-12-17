// lib/processes.ts
"use server";

import { spawn } from "child_process";

export async function execCommand<T>(cmd: string, args: string[]): Promise<T> {
  return new Promise((resolve, reject) => {
    const process = spawn(cmd, args);
    let output = "";

    process.stdout.on("data", (data) => {
      output += data.toString();
    });
    console.log(output);

    process.stderr.on("data", (data) => {
      console.error(`Error: ${data}`);
    });

    process.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(`Process exited with code ${code}`));
        return;
      }

      try {
        const res = JSON.parse(output) as T;
        resolve(res);
      } catch (error: unknown) {
        if (error instanceof Error) {
          reject(new Error(`Failed to parse JSON output: ${error.message}`));
        } else {
          reject(new Error(`Failed to parse JSON output: ${String(error)}`));
        }
      }
    });

    process.on("error", (error) => {
      reject(new Error(`Failed to start process: ${error.message}`));
    });
  });
}
