// app/api/alpaca/account/watchlists/route.ts

import { getAlpacaWatchlists } from "@/lib/alpaca";
import { NextResponse } from "next/server";


export async function GET() {
  try {
    const watchlists = await getAlpacaWatchlists();
    console.log("Watchlists fetched:", watchlists.length);
    return NextResponse.json(watchlists);
  } catch (error) {
    console.error({ data: "Watchlists fetch error: " + error });
    return NextResponse.json(
      { error: "Failed to fetch watchlists history" },
      { status: 500 },
    );
  }
}
