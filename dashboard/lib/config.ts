// lib/config.ts
export const ALPACA_OAUTH_CONFIG = {
  auth_url: "https://app.alpaca.markets/oauth/authorize",
  token_url: "https://api.alpaca.markets/oauth/token",
  redirect_uri: `${process.env.NEXT_PUBLIC_URL}/api/auth/callback`,
  scopes: "account:write trading",
};
