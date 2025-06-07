// dashboard/app/api/alpaca/positions/[symbol]/route.ts
import { NextRequest, NextResponse } from "next/server";
import { closeAlpacaPosition, getAlpacaPositions } from "@/lib/alpaca";

// Close a specific position
export async function DELETE(
  request: NextRequest,
  { params }: { params: { symbol: string } }
) {
  try {
    // Correctly get and use the symbol parameter (awaiting params)
    const symbol = params.symbol;
    
    if (!symbol) {
      return NextResponse.json(
        { error: "Symbol parameter is required" },
        { status: 400 }
      );
    }
    
    // Get position details first to get the quantity
    const positions = await getAlpacaPositions();
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