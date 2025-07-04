import { getAlpacaAccount, getAlpacaPortfolioHistory } from "@/lib/alpaca";
import { NextRequest, NextResponse } from "next/server";

export async function GET(request: NextRequest) {
  const headers = request.headers;
  const daysRaw = headers.get("days");
  let days = daysRaw ? parseInt(daysRaw) : 180;

  if (days == -1) {
    const account = await getAlpacaAccount();
    const created = account.created_at;
    const createdDate = new Date(parseInt(created)*1000);
    const now = new Date();
    const diff = now.getTime() - createdDate.getTime();
    days = diff / (1000 * 3600 * 24);
    days = Math.floor(days);
  }

  const timeframe = headers.get("timeframe") ?? "1D";

  try {
    const portfolioHistory = await getAlpacaPortfolioHistory(days, timeframe);
    console.log("Portfolio history fetched:", portfolioHistory.timestamp[-1]);
    return NextResponse.json(portfolioHistory);
  } catch (error) {
    console.error({ data: "Account history fetch error: " + error });
    return NextResponse.json(
      { error: "Failed to fetch account history" },
      { status: 500 },
    );
  }
}
