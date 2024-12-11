"use server"
import { Account } from "@alpacahq/typescript-sdk";
import { spawn } from 'child_process';

import dotenv from 'dotenv'

dotenv.config()

async function getAlpacaAccount(): Promise<Account> {
  return new Promise((resolve, reject) => {
      const process = spawn('python', ['../backend/alpaca/apca.py', 'trading/account']);
      let output = '';

      process.stdout.on('data', (data) => {
          output += data.toString();
      });

      process.stderr.on('data', (data) => {
          console.error(`Error: ${data}`);
      });

      process.on('close', (code) => {
          if (code !== 0) {
              reject(new Error(`Process exited with code ${code}`));
              return;
          }

          try {
              const account = JSON.parse(output) as Account;
              resolve(account);
          } catch (error: any) {
              reject(new Error(`Failed to parse JSON output: ${error.message}`));
          }
      });

      process.on('error', (error) => {
          reject(new Error(`Failed to start process: ${error.message}`));
      });
  });
}

export { getAlpacaAccount, type Account }