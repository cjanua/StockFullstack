import { getAlpacaWatchlists } from "@/lib/alpaca_loaders/account";
import { NextResponse } from "next/server";


export async function GET() {
  try {
    const account = await getAlpacaWatchlists();
    console.log("Account history fetched:", account);
    return NextResponse.json(account);
  } catch (error) {
    console.error({ data: "Account history fetch error: " + error });
    return NextResponse.json(
      { error: "Failed to fetch account history" },
      { status: 500 },
    );
  }
}
