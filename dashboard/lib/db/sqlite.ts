/* eslint-disable @typescript-eslint/no-unused-vars */
// lib/db/sqlite.ts
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
  id: number;
  username: string;
  email: string;
  created_at: string;
  alpaca_key?: string;
  alpaca_secret?: string;
  alpaca_auth_status?: 'active' | 'inactive' | 'error';
  last_verified?: string;
}

// Initialize database connection
let dbPromise: Promise<Database> | null = null;

export async function getDb() {
  if (!dbPromise) {
    // Ensure directory exists
    try {
      await mkdir(DB_DIR, { recursive: true });
    } catch (error) {
      // Directory already exists or cannot be created
      console.error("Error creating database directory:", error);
    }

    // Open database connection
    dbPromise = open({
      filename: DB_PATH,
      driver: sqlite3.Database,
    });

    const db = await dbPromise;
    
    // Enable foreign keys
    await db.exec('PRAGMA foreign_keys = ON');
    
    // Create tables if they don't exist
    await db.exec(`
      CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        alpaca_key TEXT,
        alpaca_secret TEXT,
        alpaca_auth_status TEXT DEFAULT 'inactive',
        last_verified TEXT
      );
      
      CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        token TEXT UNIQUE NOT NULL,
        expires_at TEXT NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
      );
    `);
  }
  
  return dbPromise;
}

// User management functions
export async function createUser(username: string, email: string, password: string): Promise<User | null> {
  const db = await getDb();
  const passwordHash = await bcrypt.hash(password, 10);
  
  try {
    const result = await db.run(
      'INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
      [username, email, passwordHash]
    );
    
    if (result.lastID) {
      const user = await db.get('SELECT id, username, email, created_at FROM users WHERE id = ?', result.lastID);
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
      'SELECT id, username, email, password_hash, created_at, alpaca_key, alpaca_secret, alpaca_auth_status, last_verified FROM users WHERE username = ? OR email = ?',
      [usernameOrEmail, usernameOrEmail]
    );
    
    if (user && await bcrypt.compare(password, user.password_hash)) {
      // Don't return the password hash
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
      'SELECT id, username, email, created_at, alpaca_key, alpaca_secret, alpaca_auth_status, last_verified FROM users WHERE id = ?',
      [id]
    );
    
    return user as User || null;
  } catch (error) {
    console.error('Error getting user by ID:', error);
    return null;
  }
}

// Alpaca token management
export async function updateAlpacaCredentials(
  userId: number, 
  alpacaKey: string, 
  alpacaSecret: string,
  authStatus: 'active' | 'inactive' | 'error' = 'active'
): Promise<boolean> {
  const db = await getDb();
  
  try {
    await db.run(
      'UPDATE users SET alpaca_key = ?, alpaca_secret = ?, alpaca_auth_status = ?, last_verified = CURRENT_TIMESTAMP WHERE id = ?',
      [alpacaKey, alpacaSecret, authStatus, userId]
    );
    return true;
  } catch (error) {
    console.error('Error updating Alpaca credentials:', error);
    return false;
  }
}

// Session management
export async function createSession(userId: number, expiresInHours = 24): Promise<string | null> {
  const db = await getDb();
  
  // Generate a cryptographically secure token
  const tokenBytes = crypto.randomBytes(32);
  const token = tokenBytes.toString('base64url'); // URL-safe base64 encoding
  
  // Calculate expiration date
  const expiresAt = new Date();
  expiresAt.setHours(expiresAt.getHours() + expiresInHours);
  
  try {
    // First, clean up any expired sessions for this user
    await db.run(
      'DELETE FROM sessions WHERE user_id = ? AND expires_at < CURRENT_TIMESTAMP',
      [userId]
    );
    
    // Limit active sessions per user (security measure)
    const activeSessions = await db.get(
      'SELECT COUNT(*) as count FROM sessions WHERE user_id = ? AND expires_at > CURRENT_TIMESTAMP',
      [userId]
    );
    
    if (activeSessions.count >= 3) {
      // Delete oldest session if user has too many active sessions
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
    
    // Create new session
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

export async function getUserBySessionToken(token: string): Promise<User | null> {
  const db = await getDb();
  
  try {
    // First check if the session is valid and not expired
    const session = await db.get(
      'SELECT * FROM sessions WHERE token = ? AND expires_at > CURRENT_TIMESTAMP',
      [token]
    );
    
    if (!session) {
      return null;
    }
    
    // Get the user associated with this session
    const user = await getUserById(session.user_id);
    return user;
  } catch (error) {
    console.error('Error getting user by session token:', error);
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