// dashboard/lib/services/alpacaAuth.ts
import { AlpacaClient } from '@/lib/alpaca-client';
import { updateAlpacaCredentials, User } from '../db/sqlite';

// Verify if Alpaca credentials are valid
export async function verifyAlpacaCredentials(
  key: string,
  secret: string,
  usePaperTrading: boolean
): Promise<boolean> {
  try {
    const client = new AlpacaClient(key, secret, usePaperTrading);
    // Try to access account info to verify credentials
    const account = await client.getAccount();
    return !!account;
  } catch (error) {
    console.error('Alpaca authentication error:', error);
    return false;
  }
}

// Store and verify Alpaca credentials for a user
export async function connectUserToAlpaca(
  userId: number,
  alpacaKey: string,
  alpacaSecret: string,
  usePaperTrading: boolean
): Promise<{ success: boolean; message: string }> {
  try {
    // Verify the credentials
    const isValid = await verifyAlpacaCredentials(alpacaKey, alpacaSecret, usePaperTrading);
    if (!isValid) {
      await updateAlpacaCredentials(userId, alpacaKey, alpacaSecret, usePaperTrading, 'error');
      return {
        success: false,
        message: 'Invalid Alpaca credentials. Please check your API key and secret in the Alpaca dashboard.',
      };
    }

    // Update the user's credentials in the database
    await updateAlpacaCredentials(userId, alpacaKey, alpacaSecret, usePaperTrading, 'active');
    return {
      success: true,
      message: 'Successfully connected to Alpaca API.',
    };
  } catch (error) {
    console.error(`Error connecting to Alpaca for user ${userId}:`, error);
    return {
      success: false,
      message: 'An error occurred while connecting to Alpaca.',
    };
  }
}

// Create an Alpaca client from user data
export async function getAlpacaClientForUser(user: User): Promise<AlpacaClient | null> {
  if (!user.alpaca_key || !user.alpaca_secret) {
    return null;
  }
  try {
    const client = new AlpacaClient(user.alpaca_key, user.alpaca_secret, user.use_paper_trading === 1);
    return client;
  } catch (error) {
    console.error(`Error creating Alpaca client for user ${user.id}:`, error);
    return null;
  }
}

// Check if user's Alpaca connection is working and update status
export async function checkAlpacaConnection(user: User): Promise<{ status: 'active' | 'inactive' | 'error'; message: string }> {
  if (!user.alpaca_key || !user.alpaca_secret) {
    return {
      status: 'inactive',
      message: 'No Alpaca credentials have been set.',
    };
  }
  try {
    const client = await getAlpacaClientForUser(user);
    if (!client) {
      await updateAlpacaCredentials(user.id, user.alpaca_key, user.alpaca_secret, user.use_paper_trading === 1, 'error');
      return {
        status: 'error',
        message: 'Failed to create Alpaca client.',
      };
    }

    // Verify connection by getting account info
    const account = await client.getAccount();
    await updateAlpacaCredentials(user.id, user.alpaca_key, user.alpaca_secret, user.use_paper_trading === 1, 'active');
    return {
      status: 'active',
      message: `Connected to Alpaca account ${account.account_number}`,
    };
  } catch (error: unknown) {
    if (!(typeof error === 'object' && error !== null)) {
      return {
        status: 'error',
        message: "Unknown Server Error",
      };
    }
    if (!(error instanceof Error)) {
      return {
        status: 'error',
        message: "Unknown Server Error",
      };
    }
    console.error(`Error checking Alpaca connection for user ${user.id}:`, error);
    const message = error.message?.includes('request is not authorized')
      ? 'Invalid Alpaca API Key or Secret. Please verify your credentials in the Alpaca dashboard.'
      : 'Failed to connect to Alpaca API. Please check your credentials.';
    await updateAlpacaCredentials(user.id, user.alpaca_key, user.alpaca_secret, user.use_paper_trading === 1, 'error');
    return {
      status: 'error',
      message,
    };
  }
}