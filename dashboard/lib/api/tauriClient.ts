// lib/api/client-wrapper.ts
import { authApi as tauriAuthApi, alpacaApi as tauriAlpacaApi } from './tauri-adapter';
import * as webAuth from './auth';
import * as webAlpaca from './alpaca';

// Detect if running in Tauri
export const isTauri = () => {
  return typeof window !== 'undefined' && window.__TAURI__ !== undefined;
};

// Export unified API that works in both environments
export const auth = isTauri() ? tauriAuthApi : webAuth;
export const alpaca = isTauri() ? tauriAlpacaApi : webAlpaca;