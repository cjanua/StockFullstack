import { Client, createClient, CreateClientOptions } from '@alpacahq/typescript-sdk';
import { updateAlpacaCredentials, User } from '../db/sqlite';
import { env } from 'process';

// Verify if Alpaca credentials are valid
export async function verifyAlpacaCredentials(key: string, secret: string, paper: boolean = false): Promise<boolean> {
  try {
    const opts: CreateClientOptions = {
      paper: paper,
      baseURL: paper ? "" : env.ALPACA_URL,
      key: key,
      secret: secret,
    };
    const alpaca: Client = createClient(opts);

    // Try to access account info to verify credentials
    const account = await alpaca.getAccount();

    // If we get here, the credentials are valid
    return true;
  } catch (error) {
    console.error("Alpaca authentication error:", error);
    return false;
  }
}

// Store and verify Alpaca credentials for a user
export async function connectUserToAlpaca(
  userId: number,
  alpacaKey: string,
  alpacaSecret: string,
  paper: boolean = false
): Promise<{ success: boolean; message: string }> {
  try {
    // First verify the credentials
    const isValid = await verifyAlpacaCredentials(alpacaKey, alpacaSecret, paper);

    if (!isValid) {
      await updateAlpacaCredentials(userId, alpacaKey, alpacaSecret, 'error');
      return {
        success: false,
        message: "Invalid Alpaca credentials. Please check your API key and secret."
      };
    }

    // Update the user's credentials in the database
    await updateAlpacaCredentials(userId, alpacaKey, alpacaSecret, 'active');

    return {
      success: true,
      message: "Successfully connected to Alpaca API."
    };
  } catch (error) {
    console.error("Error connecting to Alpaca:", error);
    return {
      success: false,
      message: "An error occurred while connecting to Alpaca."
    };
  }
}

// Create an Alpaca client from user data
export async function getAlpacaClientForUser(user: User, paper: boolean = false): Promise<Client | null> {
  if (!user.alpaca_key || !user.alpaca_secret) {
    return null;
  }

  try {
    const opts: CreateClientOptions = {
      paper: paper,
      baseURL: paper ? "" : env.ALPACA_URL,
      key: user.alpaca_key,
      secret: user.alpaca_secret,
    };
    const alpaca: Client = createClient(opts);

    return alpaca;
  } catch (error) {
    console.error("Error creating Alpaca client:", error);
    return null;
  }
}

// Check if user's Alpaca connection is working and update status
export async function checkAlpacaConnection(user: User): Promise<{ status: 'active' | 'inactive' | 'error'; message: string }> {
  if (!user.alpaca_key || !user.alpaca_secret) {
    return {
      status: 'inactive',
      message: 'No Alpaca credentials have been set.'
    };
  }

  try {
    const alpaca = await getAlpacaClientForUser(user);
    if (!alpaca) {
      return {
        status: 'error',
        message: 'Failed to create Alpaca client.'
      };
    }

    // Verify connection by getting account info
    const account = await alpaca.getAccount();

    // Update user's Alpaca status in database
    await updateAlpacaCredentials(user.id, user.alpaca_key, user.alpaca_secret, 'active');

    return {
      status: 'active',
      message: `Connected to Alpaca account ${account.account_number}`
    };
  } catch (error) {
    console.error("Error checking Alpaca connection:", error);

    // Update status to error
    await updateAlpacaCredentials(user.id, user.alpaca_key, user.alpaca_secret, 'error');

    return {
      status: 'error',
      message: 'Failed to connect to Alpaca API. Please check your credentials.'
    };
  }
}