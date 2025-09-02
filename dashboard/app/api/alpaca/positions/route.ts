// dashboard/app/api/alpaca/positions/route.ts
import { getAlpacaPositions } from "@/lib/alpaca"; // Use lib/alpaca.ts
import { getUserBySessionToken } from "@/lib/db/sqlite";
import { cookies } from "next/headers";
import { NextResponse } from "next/server";

export async function GET(): Promise<NextResponse> {
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
    const positions = await getAlpacaPositions(user.id.toString());
    return NextResponse.json(positions, { status: 200 });
  } catch (error: unknown) {
    if (!(typeof error === 'object' && error !== null)) {
      console.error(`Unexpected error fetching account for user ${user.id}:`, error);
      return NextResponse.json({ error: "Unknown Server Error" }, { status: 500 });
    }
    if (!(error instanceof Error)) {
      console.error(`Unexpected error fetching account for user ${user.id}:`, error);
      return NextResponse.json({ error: "Unknown Server Error" }, { status: 500 });
    }
    console.error(`Error fetching positions for user ${user.id}:`, error);
    if (error.message.includes("request is not authorized")) {
      return NextResponse.json({ error: "Invalid Alpaca credentials" }, { status: 401 });
    }
    if (error.message.includes("User Alpaca credentials not configured")) {
      return NextResponse.json({ error: "Alpaca credentials not set" }, { status: 400 });
    }
    return NextResponse.json({ error: "Internal Server Error" }, { status: 500 });
  }
}