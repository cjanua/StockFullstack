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
    
    // Add validation for required fields
    if (!body.symbol) {
      return NextResponse.json(
        { error: "Missing required field: symbol" },
        { status: 400 }
      );
    }
    
    if (!body.qty && !body.notional) {
      return NextResponse.json(
        { error: "Either qty or notional must be provided" },
        { status: 400 }
      );
    }
    
    if (!body.side || !['buy', 'sell'].includes(body.side.toLowerCase())) {
      return NextResponse.json(
        { error: "Side must be either 'buy' or 'sell'" },
        { status: 400 }
      );
    }
    
    // Log the order request for debugging
    console.log(`Creating order: ${JSON.stringify(body, null, 2)}`);
    
    try {
      // Pass the request body directly to the Alpaca client
      const order = await createAlpacaOrder(body);
      
      // After order is placed, clear the backend cache
      try {
        await fetch('http://localhost:8001/api/portfolio/clear-cache', {
          method: 'POST',
          cache: 'no-store',
        });
      } catch (cacheError) {
        console.error('Failed to clear cache:', cacheError);
      }
      
      return NextResponse.json(order, { status: 201 });
      
    } catch (alpacaError) {
      // Handle specific Alpaca API errors
      console.error("Alpaca API error:", alpacaError);
      
      // Format alpaca error for better debugging and client response
      const errorDetails = formatAlpacaError(alpacaError);
      
      return NextResponse.json(
        { 
          error: "Order rejected by Alpaca", 
          details: errorDetails 
        },
        { status: 422 } // Unprocessable Entity - order was understood but rejected
      );
    }
  } catch (error) {
    console.error("Order placement error:", error);
    return NextResponse.json(
      { 
        error: "Failed to place order", 
        details: error instanceof Error ? error.message : "Unknown error" 
      },
      { status: 500 }
    );
  }
}

// Helper function to format Alpaca error messages
function formatAlpacaError(error: unknown): string {
  if (error instanceof Error) {
    const errorMsg = error.message;
    
    // Common Alpaca API error patterns to make more user-friendly
    if (errorMsg.includes("insufficient buying power")) {
      return "Insufficient funds to execute this order.";
    }
    
    if (errorMsg.includes("position is not found")) {
      return "You don't have a position in this security.";
    }
    
    // Check for JSON-formatted error messages
    try {
      if (errorMsg.includes("{") && errorMsg.includes("}")) {
        const jsonStart = errorMsg.indexOf('{');
        const errorObj = JSON.parse(errorMsg.slice(jsonStart));
        if (errorObj.message) {
          return errorObj.message;
        }
      }
    } catch (e) {
      // Parsing failed, continue with normal error handling
    }
    
    return errorMsg;
  }
  
  return "Unknown error occurred while processing your order";
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