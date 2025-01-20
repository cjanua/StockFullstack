import { getAlpacaPositions } from "@/lib/alpaca";
import { NextResponse } from "next/server";

export async function GET() {
  try {
    // const user = await verifySession();
    const ps = await getAlpacaPositions();

    console.log("Positions fetched:", ps);
    const positions = ps.sort((a, b) =>
      parseFloat(a.qty) * parseFloat(a.current_price) >
      parseFloat(b.qty) * parseFloat(b.current_price)
        ? -1
        : 1,
    );
    return NextResponse.json(positions);
  } catch (error) {
    console.error({ data: "Positions fetch error: " + error });
    return NextResponse.json(
      { error: "Failed to fetch positions" },
      { status: 500 },
    );
  }
}
