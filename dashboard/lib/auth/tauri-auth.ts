// lib/auth/tauri-auth.ts
import { Store } from 'tauri-plugin-store-api';

const store = new Store('.auth.dat');

export async function saveAuthToken(token: string) {
  await store.set('auth_token', token);
  await store.save();
}

export async function getAuthToken(): Promise<string | null> {
  return await store.get('auth_token');
}

export async function clearAuthToken() {
  await store.delete('auth_token');
  await store.save();
}