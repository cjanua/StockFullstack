// dashboard/app/api/alpaca/positions/route.ts
import { NextRequest, NextResponse } from "next/server";
import { getAlpacaPositions } from "@/lib/alpaca";

// Get all positions
export async function GET(_request: NextRequest) {
  try {
    const positions = await getAlpacaPositions();
    return NextResponse.json(positions);
  } catch (error) {
    console.error("Error getting positions:", error);
    return NextResponse.json(
      { error: "Failed to fetch positions" },
      { status: 500 }
    );
  }
}