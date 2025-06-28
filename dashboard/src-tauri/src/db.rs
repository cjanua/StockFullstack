use rusqlite::{Connection, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use bcrypt::{hash, verify, DEFAULT_COST};
use uuid::Uuid;
use chrono::{DateTime, Utc, Duration};

#[derive(Debug, Serialize, Deserialize)]
pub struct User {
    pub id: i32,
    pub username: String,
    pub email: String,
    pub created_at: String,
    pub alpaca_key: Option<String>,
    pub alpaca_secret: Option<String>,
    pub alpaca_auth_status: Option<String>,
    pub last_verified: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Session {
    pub id: i32,
    pub user_id: i32,
    pub token: String,
    pub expires_at: String,
    pub created_at: String,
}

pub struct Database {
    conn: Connection,
}

impl Database {
    pub fn new(db_path: &Path) -> Result<Self> {
        let conn = Connection::open(db_path)?;
        
        // Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON", [])?;
        
        // Create tables
        conn.execute(
            "CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                alpaca_key TEXT,
                alpaca_secret TEXT,
                alpaca_auth_status TEXT DEFAULT 'inactive',
                last_verified TEXT
            )",
            [],
        )?;
        
        conn.execute(
            "CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                token TEXT UNIQUE NOT NULL,
                expires_at TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )",
            [],
        )?;
        
        Ok(Database { conn })
    }
    
    pub fn verify_user(&self, username_or_email: &str, password: &str) -> Result<Option<User>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, username, email, password_hash, created_at, alpaca_key, 
             alpaca_secret, alpaca_auth_status, last_verified 
             FROM users WHERE username = ?1 OR email = ?1"
        )?;
        
        let user_result = stmt.query_row([username_or_email], |row| {
            let password_hash: String = row.get(3)?;
            
            Ok((
                User {
                    id: row.get(0)?,
                    username: row.get(1)?,
                    email: row.get(2)?,
                    created_at: row.get(4)?,
                    alpaca_key: row.get(5)?,
                    alpaca_secret: row.get(6)?,
                    alpaca_auth_status: row.get(7)?,
                    last_verified: row.get(8)?,
                },
                password_hash
            ))
        });
        
        match user_result {
            Ok((user, hash)) => {
                if verify(password, &hash).unwrap_or(false) {
                    Ok(Some(user))
                } else {
                    Ok(None)
                }
            }
            Err(_) => Ok(None),
        }
    }
    
    pub fn create_session(&self, user_id: i32, expires_in_hours: i64) -> Result<String> {
        let token = Uuid::new_v4().to_string();
        let expires_at = Utc::now() + Duration::hours(expires_in_hours);
        
        self.conn.execute(
            "INSERT INTO sessions (user_id, token, expires_at) VALUES (?1, ?2, ?3)",
            [&user_id.to_string(), &token, &expires_at.to_rfc3339()],
        )?;
        
        Ok(token)
    }
    
    pub fn get_user_by_token(&self, token: &str) -> Result<Option<User>> {
        let mut stmt = self.conn.prepare(
            "SELECT u.id, u.username, u.email, u.created_at, u.alpaca_key, 
             u.alpaca_secret, u.alpaca_auth_status, u.last_verified
             FROM users u
             JOIN sessions s ON u.id = s.user_id
             WHERE s.token = ?1 AND s.expires_at > datetime('now')"
        )?;
        
        let user_result = stmt.query_row([token], |row| {
            Ok(User {
                id: row.get(0)?,
                username: row.get(1)?,
                email: row.get(2)?,
                created_at: row.get(3)?,
                alpaca_key: row.get(4)?,
                alpaca_secret: row.get(5)?,
                alpaca_auth_status: row.get(6)?,
                last_verified: row.get(7)?,
            })
        });
        
        match user_result {
            Ok(user) => Ok(Some(user)),
            Err(_) => Ok(None),
        }
    }
    
    pub fn update_alpaca_credentials(
        &self,
        user_id: i32,
        alpaca_key: &str,
        alpaca_secret: &str,
        status: &str,
    ) -> Result<()> {
        self.conn.execute(
            "UPDATE users SET alpaca_key = ?1, alpaca_secret = ?2, 
             alpaca_auth_status = ?3, last_verified = datetime('now') 
             WHERE id = ?4",
            [alpaca_key, alpaca_secret, status, &user_id.to_string()],
        )?;
        Ok(())
    }
}