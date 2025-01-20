// lib/processes.ts
"use server";

import { spawn } from "child_process";

export async function execCommand<T>(cmd: string, args: string[]): Promise<T> {
  return new Promise((resolve, reject) => {
    console.log(`Executing command: ${cmd} ${args.join(" ")}`); // Debug log

    const process = spawn(cmd, args);
    let stdout = "";
    let stderr = "";

    process.stdout.on("data", (data) => {
      stdout += data.toString();
      console.log(`stdout: ${data}`); // Real-time stdout logging
    });

    process.stderr.on("data", (data) => {
      stderr += data.toString();
      console.info(`stderr: ${data}`); // Real-time stderr logging
    });

    process.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(`Process exited with code ${code}. Error: ${stderr}`));
        return;
      }

      if (!stdout.trim()) {
        reject(new Error("No output received from process"));
        return;
      }

      try {
        // Clean the output before parsing
        const cleanOutput = stdout.trim();
        if (!cleanOutput) {
          reject(new Error("No output received"));
          return;
        }

        const parsed = JSON.parse(cleanOutput) as T;

        // Check for error structure
        if (
          typeof parsed === "object" &&
          parsed !== null &&
          "error" in parsed
        ) {
          const errorObject = parsed as { message?: string };
          reject(new Error(errorObject.message || "Unknown error"));
          return;
        }

        resolve(parsed);
      } catch (error) {
        reject(new Error(`JSON parse error: ${error}\nRaw output: ${stdout}`));
      }
    });

    process.on("error", (error) => {
      console.error(`Process error: ${error.message}`); // Debug log
      reject(new Error(`Failed to start process: ${error.message}`));
    });

    // Handle process timeout
    const timeout = setTimeout(() => {
      process.kill();
      reject(new Error("Process timed out after 30 seconds"));
    }, 30000);

    // Clear timeout on success or error
    process.on("close", () => clearTimeout(timeout));
  });
}
