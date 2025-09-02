// dashboard/app/api/alpaca/quotes/route.ts

import { getAlpacaLatestQuotes } from "@/lib/alpaca";
import { getUserBySessionToken } from "@/lib/db/sqlite";
import { cookies } from "next/headers";
import { NextResponse } from "next/server";

export async function GET(request: Request) {
  const cookieStore = await cookies();
  const authToken = cookieStore.get("auth_token")?.value;

  if (!authToken) {
    return NextResponse.json({ error: "Not authenticated" }, { status: 401 });
  }

  const user = await getUserBySessionToken(authToken);
  if (!user) {
    return NextResponse.json({ error: "Invalid session" }, { status: 401 });
  }

  const { searchParams } = new URL(request.url);
  const symbolsParam = searchParams.get("symbols");

  if (!symbolsParam) {
    return NextResponse.json({ error: "Symbols are required" }, { status: 400 });
  }

  const symbols = symbolsParam.split(",");

  try {
    const quotes = await getAlpacaLatestQuotes(user.id.toString(), symbols);

    // Convert the array of quotes into a map for easy lookup on the client
    const quotesMap = quotes.reduce((acc: any, quote: any) => {
      acc[quote.symbol] = quote;
      return acc;
    }, {});

    return NextResponse.json(quotesMap, { status: 200 });
  } catch (error: any) {
    console.error(`Error fetching batched quotes for user ${user.id}:`, error);
    return NextResponse.json(
      { error: "Failed to fetch quotes" },
      { status: 500 }
    );
  }
}