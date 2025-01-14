import { getAlpacaAccount } from "@/lib/alpaca";
import { verifySession } from "@/lib/auth";
import { NextResponse } from "next/server";

export async function GET() {
  try {
    // const user = await verifySession();
    const account = await getAlpacaAccount();
    console.log("Account fetched:", account);
    return NextResponse.json(account);
  } catch (error) {
    console.error({data: ( "Account fetch error: " + error)});
    return NextResponse.json(
      { error: "Failed to fetch account" },
      { status: 500 },
    );
  }
}
