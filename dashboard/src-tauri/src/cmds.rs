use crate::database::{Database, User};
use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use tauri::State;

#[derive(Debug, Serialize, Deserialize)]
pub struct LoginRequest {
    username: String,
    password: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LoginResponse {
    success: bool,
    user: Option<User>,
    token: Option<String>,
    error: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AlpacaCredentials {
    alpaca_key: String,
    alpaca_secret: String,
    paper: bool,
}

pub struct AppState {
    pub db: Mutex<Database>,
}

#[tauri::command]
pub async fn login(
    state: State<'_, AppState>,
    credentials: LoginRequest,
) -> Result<LoginResponse, String> {
    let db = state.db.lock().unwrap();
    
    match db.verify_user(&credentials.username, &credentials.password) {
        Ok(Some(user)) => {
            match db.create_session(user.id, 24) {
                Ok(token) => Ok(LoginResponse {
                    success: true,
                    user: Some(user),
                    token: Some(token),
                    error: None,
                }),
                Err(e) => Ok(LoginResponse {
                    success: false,
                    user: None,
                    token: None,
                    error: Some(format!("Failed to create session: {}", e)),
                }),
            }
        }
        Ok(None) => Ok(LoginResponse {
            success: false,
            user: None,
            token: None,
            error: Some("Invalid username or password".to_string()),
        }),
        Err(e) => Ok(LoginResponse {
            success: false,
            user: None,
            token: None,
            error: Some(format!("Database error: {}", e)),
        }),
    }
}

#[tauri::command]
pub async fn get_current_user(
    state: State<'_, AppState>,
    token: String,
) -> Result<Option<User>, String> {
    let db = state.db.lock().unwrap();
    
    match db.get_user_by_token(&token) {
        Ok(user) => Ok(user),
        Err(e) => Err(format!("Database error: {}", e)),
    }
}

#[tauri::command]
pub async fn update_alpaca_credentials(
    state: State<'_, AppState>,
    user_id: i32,
    credentials: AlpacaCredentials,
) -> Result<bool, String> {
    let db = state.db.lock().unwrap();
    
    // TODO: Verify credentials with Alpaca API
    
    match db.update_alpaca_credentials(
        user_id,
        &credentials.alpaca_key,
        &credentials.alpaca_secret,
        "active",
    ) {
        Ok(_) => Ok(true),
        Err(e) => Err(format!("Failed to update credentials: {}", e)),
    }
}

// Add proxy command for Alpaca API calls
#[tauri::command]
pub async fn alpaca_api_call(
    endpoint: String,
    method: String,
    body: Option<String>,
    alpaca_key: String,
    alpaca_secret: String,
) -> Result<String, String> {
    let client = reqwest::Client::new();
    let url = format!("https://api.alpaca.markets{}", endpoint);
    
    let mut request = match method.as_str() {
        "GET" => client.get(&url),
        "POST" => client.post(&url),
        "PUT" => client.put(&url),
        "DELETE" => client.delete(&url),
        _ => return Err("Invalid HTTP method".to_string()),
    };
    
    request = request
        .header("APCA-API-KEY-ID", alpaca_key)
        .header("APCA-API-SECRET-KEY", alpaca_secret);
    
    if let Some(body_str) = body {
        request = request
            .header("Content-Type", "application/json")
            .body(body_str);
    }
    
    match request.send().await {
        Ok(response) => {
            match response.text().await {
                Ok(text) => Ok(text),
                Err(e) => Err(format!("Failed to read response: {}", e)),
            }
        }
        Err(e) => Err(format!("Request failed: {}", e)),
    }
}