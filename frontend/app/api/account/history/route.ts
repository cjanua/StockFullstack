import { getAlpacaAccountHistory } from "@/lib/alpaca";
import { NextRequest, NextResponse } from "next/server";

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const days = parseInt(searchParams.get("days") ?? '30');

  try {
    const account = await getAlpacaAccountHistory(days);
    console.log("Account fetched:", account);
    return NextResponse.json(account);
  } catch (error) {
    console.error({ data: "Account fetch error: " + error });
    return NextResponse.json(
      { error: "Failed to fetch account" },
      { status: 500 },
    );
  }
}
