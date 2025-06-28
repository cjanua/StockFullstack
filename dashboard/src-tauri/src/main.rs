// Prevents additional console window on Windows in release, DO NOT REMOVE!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod database;

use commands::{AppState, login, get_current_user, update_alpaca_credentials, alpaca_api_call};
use database::Database;
use std::sync::Mutex;
use tauri::Manager;

fn main() {
    tauri::Builder::default()
        .setup(|app| {
            let app_dir = app.path_resolver().app_data_dir().unwrap();
            std::fs::create_dir_all(&app_dir).unwrap();
            
            let db_path = app_dir.join("auth.db");
            let db = Database::new(&db_path).expect("Failed to create database");
            
            app.manage(AppState {
                db: Mutex::new(db),
            });
            
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            login,
            get_current_user,
            update_alpaca_credentials,
            alpaca_api_call
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}