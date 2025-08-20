// dashboard/lib/db/sqlite.ts (partial update)
import sqlite3 from 'sqlite3';
import { open, Database } from 'sqlite';
import { join } from 'path';
import { mkdir } from 'fs/promises';
import bcrypt from 'bcryptjs';
import crypto from 'crypto';

// Define database directory and path
const DB_DIR = join(process.cwd(), 'data');
const DB_PATH = join(DB_DIR, 'auth.db');

// Interface definitions
export interface User {
  id: number; // Kept as number for auto-incrementing integer
  username: string;
  email: string;
  created_at: string;
  alpaca_key?: string;
  alpaca_secret?: string;
  alpaca_auth_status?: 'active' | 'inactive' | 'error';
  last_verified?: string;
  use_paper_trading?: number; // Added to support existing column
}

// Initialize database connection
let dbPromise: Promise<Database> | null = null;
export async function getDb() {
  if (!dbPromise) {
    try {
      await mkdir(DB_DIR, { recursive: true });
    } catch (error) {
      console.error('Error creating database directory:', error);
    }
    dbPromise = open({
      filename: DB_PATH,
      driver: sqlite3.Database,
    });
    const db = await dbPromise;
    await db.exec('PRAGMA foreign_keys = ON');
    // Schema unchanged as per instruction
  }
  return dbPromise;
}

// Update Alpaca credentials
export async function updateAlpacaCredentials(
  userId: number,
  alpacaKey: string,
  alpacaSecret: string,
  usePaperTrading: boolean,
  authStatus: 'active' | 'inactive' | 'error' = 'active'
): Promise<boolean> {
  const db = await getDb();
  try {
    await db.run(
      'UPDATE users SET alpaca_key = ?, alpaca_secret = ?, use_paper_trading = ?, alpaca_auth_status = ?, last_verified = CURRENT_TIMESTAMP WHERE id = ?',
      [alpacaKey, alpacaSecret, usePaperTrading ? 1 : 0, authStatus, userId]
    );
    return true;
  } catch (error) {
    console.error(`Error updating Alpaca credentials for user ${userId}:`, error);
    return false;
  }
}

// Existing functions (updated to include use_paper_trading)
export async function createUser(username: string, email: string, password: string): Promise<User | null> {
  const db = await getDb();
  const passwordHash = await bcrypt.hash(password, 10);
  try {
    const result = await db.run(
      'INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
      [username, email, passwordHash]
    );
    if (result.lastID) {
      const user = await db.get(
        'SELECT id, username, email, created_at, use_paper_trading FROM users WHERE id = ?',
        result.lastID
      );
      return user as User;
    }
  } catch (error) {
    console.error('Error creating user:', error);
  }
  return null;
}

export async function verifyUser(usernameOrEmail: string, password: string): Promise<User | null> {
  const db = await getDb();
  try {
    const user = await db.get(
      'SELECT id, username, email, password_hash, created_at, alpaca_key, alpaca_secret, alpaca_auth_status, last_verified, use_paper_trading FROM users WHERE username = ? OR email = ?',
      [usernameOrEmail, usernameOrEmail]
    );
    if (user && (await bcrypt.compare(password, user.password_hash))) {
      const { password_hash, ...userWithoutPassword } = user;
      return userWithoutPassword as User;
    }
  } catch (error) {
    console.error('Error verifying user:', error);
  }
  return null;
}

export async function getUserById(id: number): Promise<User | null> {
  const db = await getDb();
  try {
    const user = await db.get(
      'SELECT id, username, email, created_at, alpaca_key, alpaca_secret, alpaca_auth_status, last_verified, use_paper_trading FROM users WHERE id = ?',
      [id]
    );
    return user as User | null;
  } catch (error) {
    console.error('Error getting user by ID:', error);
    return null;
  }
}

export async function getUserBySessionToken(token: string): Promise<User | null> {
  const db = await getDb();
  try {
    const session = await db.get(
      'SELECT * FROM sessions WHERE token = ? AND expires_at > CURRENT_TIMESTAMP',
      [token]
    );
    if (!session) {
      return null;
    }
    const user = await getUserById(session.user_id);
    return user;
  } catch (error) {
    console.error('Error getting user by session token:', error);
    return null;
  }
}

export async function createSession(userId: number, expiresInHours = 24): Promise<string | null> {
  const db = await getDb();
  const tokenBytes = crypto.randomBytes(32);
  const token = tokenBytes.toString('base64url');
  const expiresAt = new Date();
  expiresAt.setHours(expiresAt.getHours() + expiresInHours);
  try {
    await db.run(
      'DELETE FROM sessions WHERE user_id = ? AND expires_at < CURRENT_TIMESTAMP',
      [userId]
    );
    const activeSessions = await db.get(
      'SELECT COUNT(*) as count FROM sessions WHERE user_id = ? AND expires_at > CURRENT_TIMESTAMP',
      [userId]
    );
    if (activeSessions.count >= 3) {
      await db.run(
        `DELETE FROM sessions
         WHERE user_id = ?
         AND id = (
           SELECT id FROM sessions
           WHERE user_id = ?
           ORDER BY created_at ASC
           LIMIT 1
         )`,
        [userId, userId]
      );
    }
    await db.run(
      'INSERT INTO sessions (user_id, token, expires_at) VALUES (?, ?, ?)',
      [userId, token, expiresAt.toISOString()]
    );
    console.log(`[SECURITY] Session created for user ${userId}, expires at ${expiresAt.toISOString()}`);
    return token;
  } catch (error) {
    console.error('Error creating session:', error);
    return null;
  }
}

export async function deleteSession(token: string): Promise<boolean> {
  const db = await getDb();
  try {
    await db.run('DELETE FROM sessions WHERE token = ?', [token]);
    return true;
  } catch (error) {
    console.error('Error deleting session:', error);
    return false;
  }
}
