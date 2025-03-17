// frontend/app/api/alpaca/orders/route.ts
import { NextRequest, NextResponse } from "next/server";
import { 
  createAlpacaOrder, 
  getAlpacaOrders, 
  cancelAlpacaOrder, 
  cancelAllAlpacaOrders
} from "@/lib/alpaca";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    
    // Pass the request body directly to the Alpaca client
    // This preserves all the field names as expected by the Alpaca API
    const order = await createAlpacaOrder(body);
    
    return NextResponse.json(order, { status: 201 });
  } catch (error) {
    console.error("Order placement error:", error);
    return NextResponse.json(
      { error: "Failed to place order", details: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}

export async function GET(request: NextRequest) {
  try {
    // Get query parameters
    const searchParams = request.nextUrl.searchParams;
    const params: any = {};
    
    // Add parameters if present
    if (searchParams.has('status')) {
      params.status = searchParams.get('status');
    }
    
    if (searchParams.has('limit')) {
      params.limit = parseInt(searchParams.get('limit')!);
    }
    
    if (searchParams.has('direction')) {
      params.direction = searchParams.get('direction');
    }
    
    if (searchParams.has('after')) {
      params.after = new Date(searchParams.get('after')!);
    }
    
    if (searchParams.has('until')) {
      params.until = new Date(searchParams.get('until')!);
    }
    
    // Get orders
    const orders = await getAlpacaOrders(params);
    
    return NextResponse.json(orders);
  } catch (error) {
    console.error("Orders fetch error:", error);
    return NextResponse.json(
      { error: "Failed to fetch orders", details: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}

export async function DELETE(request: NextRequest) {
  try {
    const orderId = request.nextUrl.searchParams.get('id');
    
    if (!orderId) {
      // If no order ID is provided, cancel all open orders
      await cancelAllAlpacaOrders();
      return NextResponse.json({ message: "All orders canceled" });
    } else {
      // Cancel specific order
      await cancelAlpacaOrder(orderId);
      return NextResponse.json({ message: `Order ${orderId} canceled` });
    }
  } catch (error) {
    console.error("Order cancellation error:", error);
    return NextResponse.json(
      { error: "Failed to cancel order", details: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}