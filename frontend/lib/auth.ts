// lib/auth.ts
import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import prisma from "./userdb";
import type { AlpacaUser } from "../types/auth";

const ALPACA_CLIENT_ID = process.env.ALPACA_CLIENT_ID!;
const ALPACA_CLIENT_SECRET = process.env.ALPACA_CLIENT_SECRET!;
const REDIRECT_URI = process.env.NEXT_PUBLIC_URL! + "/api/auth/callback";


const formatClientId = (id: string): string => {
  // Remove any non-alphanumeric characters
  const cleaned = id.replace(/[^a-zA-Z0-9]/g, '');
  
  // Ensure we have enough characters (32)
  if (cleaned.length < 32) {
      throw new Error('Client ID must be at least 32 characters');
  }
  
  // Format as UUID style (8-4-4-4-12)
  const parts = [
      cleaned.slice(0, 8),
      cleaned.slice(8, 12),
      cleaned.slice(12, 16),
      cleaned.slice(16, 20),
      cleaned.slice(20, 32)
  ];
  
  return parts.join('-');
}

export const getAlpacaAuthUrl = async () => {
  const state = crypto.randomUUID();

  const cookieStore = await cookies();
  cookieStore.set("auth_state", state, {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
  });

  return (
    `https://app.alpaca.markets/oauth/authorize?` +
    `response_type=code&` +
    `client_id=${(formatClientId(ALPACA_CLIENT_ID))}&` +
    `redirect_uri=${encodeURIComponent(REDIRECT_URI)}&` +
    `state=${state}&` +
    `scope=account:write%20trading`
  );
};

export async function verifySession(): Promise<AlpacaUser> {
  const sessionToken = (await cookies()).get("session_token")?.value;

  if (!sessionToken) redirect("/api/auth/login");

  const user = await prisma.user.findUnique({
    where: { sessionToken },
  });

  if (!user) redirect("/api/auth/login");

  return user;
}
