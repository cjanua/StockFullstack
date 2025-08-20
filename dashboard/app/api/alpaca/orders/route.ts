// dashboard/app/api/alpaca/orders/route.ts
import { getUserBySessionToken } from "@/lib/db/sqlite";
import { createAlpacaOrder, getAlpacaOrders, cancelAlpacaOrder, cancelAllAlpacaOrders } from "@/lib/alpaca"; // Use lib/alpaca.ts
import { cookies } from "next/headers";
import { NextResponse } from "next/server";

export async function GET(request: Request): Promise<NextResponse> {
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
    const url = new URL(request.url);
    const params = Object.fromEntries(url.searchParams);
    const orders = await getAlpacaOrders(user.id.toString(), params);
    return NextResponse.json(orders, { status: 200 });
  } catch (error: any) {
    console.error(`Error fetching orders for user ${user.id}:`, error);
    if (error.message.includes("request is not authorized")) {
      return NextResponse.json({ error: "Invalid Alpaca credentials" }, { status: 401 });
    }
    if (error.message.includes("User Alpaca credentials not configured")) {
      return NextResponse.json({ error: "Alpaca credentials not set" }, { status: 400 });
    }
    return NextResponse.json({ error: "Internal Server Error" }, { status: 500 });
  }
}

export async function POST(request: Request): Promise<NextResponse> {
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
    const orderData = await request.json();
    const order = await createAlpacaOrder(user.id.toString(), orderData);
    return NextResponse.json(order, { status: 200 });
  } catch (error: any) {
    console.error(`Error creating order for user ${user.id}:`, error);
    if (error.message.includes("request is not authorized")) {
      return NextResponse.json({ error: "Invalid Alpaca credentials" }, { status: 401 });
    }
    if (error.message.includes("User Alpaca credentials not configured")) {
      return NextResponse.json({ error: "Alpaca credentials not set" }, { status: 400 });
    }
    return NextResponse.json({ error: "Internal Server Error" }, { status: 500 });
  }
}

export async function DELETE(request: Request): Promise<NextResponse> {
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
    const url = new URL(request.url);
    const orderId = url.searchParams.get("id");
    if (orderId) {
      const result = await cancelAlpacaOrder(user.id.toString(), orderId);
      return NextResponse.json(result, { status: 200 });
    } else {
      await cancelAllAlpacaOrders(user.id.toString());
      return NextResponse.json({ success: true }, { status: 200 });
    }
  } catch (error: any) {
    console.error(`Error canceling order(s) for user ${user.id}:`, error);
    if (error.message.includes("request is not authorized")) {
      return NextResponse.json({ error: "Invalid Alpaca credentials" }, { status: 401 });
    }
    if (error.message.includes("User Alpaca credentials not configured")) {
      return NextResponse.json({ error: "Alpaca credentials not set" }, { status: 400 });
    }
    return NextResponse.json({ error: "Internal Server Error" }, { status: 500 });
  }
}
