// dashboard/app/api/alpaca/connect/route.ts
import { getUserBySessionToken } from "@/lib/db/sqlite";
import Database from "better-sqlite3";
import path from "path";
import { cookies } from "next/headers";
import { NextResponse } from "next/server";

export async function POST(request: Request): Promise<NextResponse> {
  const cookieStore = await cookies();
  const authToken = cookieStore.get("auth_token")?.value;

  if (!authToken) {
    return NextResponse.json({ error: "Not authenticated" }, { status: 401 });
  }

  const user = await getUserBySessionToken(authToken);
  if (!user) {
    return NextResponse.json({ error: "Invalid or expired session" }, { status: 401 });
  }

  try {
    const { alpaca_key, alpaca_secret, usePaperTrading } = await request.json();
    if (!alpaca_key || !alpaca_secret) {
      return NextResponse.json({ error: "Missing keys" }, { status: 400 });
    }

    const dbPath = path.resolve(process.cwd(), "data/auth.db");
    const db = new Database(dbPath, { readonly: false });
    try {
      const stmt = db.prepare(
        "UPDATE users SET alpaca_key = ?, alpaca_secret = ?, use_paper_trading = ? WHERE id = ?"
      );
      stmt.run(alpaca_key, alpaca_secret, usePaperTrading ? 1 : 0, user.id);
    } finally {
      db.close();
    }

    return NextResponse.json({ success: true }, { status: 200 });
  } catch (error) {
    console.error(`Error saving Alpaca keys for user ${user.id}:`, error);
    return NextResponse.json({ error: "Internal Server Error" }, { status: 500 });
  }
}