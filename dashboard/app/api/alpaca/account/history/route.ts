// dashboard/app/api/alpaca/account/history/route.ts
import { getAlpacaPortfolioHistory } from "@/lib/alpaca"; // Use lib/alpaca.ts
import { getUserBySessionToken } from "@/lib/db/sqlite";
import { cookies } from "next/headers";
import { NextResponse } from "next/server";

export async function GET(request: Request): Promise<NextResponse> {
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
    const url = new URL(request.url);
    const days = parseInt(url.searchParams.get("days") || "7", 10);
    const timeframe = url.searchParams.get("timeframe") || "1D";
    const history = await getAlpacaPortfolioHistory(user.id.toString(), days, timeframe);
    return NextResponse.json(history, { status: 200 });
  } catch (error: any) {
    console.error(`Error fetching portfolio history for user ${user.id}:`, error);
    if (error.message.includes("request is not authorized")) {
      return NextResponse.json({ error: "Invalid Alpaca credentials" }, { status: 401 });
    }
    if (error.message.includes("User Alpaca credentials not configured")) {
      return NextResponse.json({ error: "Alpaca credentials not set" }, { status: 400 });
    }
    return NextResponse.json({ error: "Internal Server Error" }, { status: 500 });
  }
}