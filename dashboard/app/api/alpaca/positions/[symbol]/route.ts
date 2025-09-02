// dashboard/app/api/alpaca/positions/[symbol]/route.ts
import { NextRequest, NextResponse } from "next/server";
import { closeAlpacaPosition, getAlpacaPositions } from "@/lib/alpaca";
import { cookies } from "next/headers";
import { getUserBySessionToken } from "@/lib/db/sqlite";

// Close a specific position
export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ symbol: string }> }
) {
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
    const resolvedParams = await params;
    const symbol = resolvedParams.symbol;

    if (!symbol) {
      return NextResponse.json(
        { error: "Symbol parameter is required" },
        { status: 400 }
      );
    }

    // Get position details first to get the quantity
    const positions = await getAlpacaPositions(user.id.toString());
    const position = positions.find(p => p.symbol.toLowerCase() === symbol.toLowerCase());

    if (!position) {
      return NextResponse.json(
        { error: `Position for ${symbol} not found` },
        { status: 404 }
      );
    }

    // Close the position - explicitly passing the quantity
    await closeAlpacaPosition(symbol, position.qty, position.side === 'long' ? 'long' : 'short');

    return NextResponse.json(
      { message: `Position ${symbol} closed successfully` }
    );
  } catch (error) {
    console.error(`Error closing position:`, error);
    return NextResponse.json(
      {
        error: "Failed to close position",
        details: error instanceof Error ? error.message : "Unknown error"
      },
      { status: 500 }
    );
  }
}