// frontend/app/api/alpaca/positions/route.ts
import { NextRequest, NextResponse } from "next/server";
import { getAlpacaPositions, closeAlpacaPosition } from "@/lib/alpaca";

// Close all positions
// export async function DELETE(request: NextRequest) {
//   try {
//     // Get all open positions
//     const positions = await getAlpacaPositions();
    
//     if (!positions || positions.length === 0) {
//       return NextResponse.json({ message: "No open positions to close" });
//     }
    
//     // Close each position one by one
//     const closePromises = positions.map(position => 
//       closeAlpacaPosition(position.symbol)
//     );
    
//     // Wait for all positions to be closed
//     await Promise.all(closePromises);
    
//     return NextResponse.json({
//       message: `Successfully closed ${positions.length} positions`
//     });
//   } catch (error) {
//     console.error("Error closing all positions:", error);
//     return NextResponse.json(
//       { 
//         error: "Failed to close all positions", 
//         details: error instanceof Error ? error.message : "Unknown error" 
//       },
//       { status: 500 }
//     );
//   }
// }

// Get all positions
export async function GET(request: NextRequest) {
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