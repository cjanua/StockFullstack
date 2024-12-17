// app/api/auth/login/route.ts
import { ALPACA_OAUTH_CONFIG } from "@/lib/config";
import { cookies } from "next/headers";
import { redirect } from "next/navigation";

export async function GET() {
  const state = crypto.randomUUID();
  const cookieStore = await cookies();
  cookieStore.set("auth_state", state, {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
  });

  const authUrl =
    `${ALPACA_OAUTH_CONFIG.auth_url}?` +
    `response_type=code&` +
    `client_id=${process.env.ALPACA_CLIENT_ID}&` +
    `redirect_uri=${encodeURIComponent(ALPACA_OAUTH_CONFIG.redirect_uri)}&` +
    `state=${state}&` +
    `scope=${ALPACA_OAUTH_CONFIG.scopes}`;

  redirect(authUrl);
}
