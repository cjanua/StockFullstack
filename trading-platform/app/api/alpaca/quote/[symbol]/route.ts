import { getAlpacaLatestQuote } from '@/lib/alpaca';
import { NextRequest, NextResponse } from 'next/server';

// Define the expected structure for route parameters
interface RouteParams {
  symbol: string;
}

export async function GET(
  request: NextRequest,
  context: { params: RouteParams } // Use context object directly
) {
  // Await the context.params object before accessing its properties
  const params = await context.params;
  const symbol = params.symbol;
  if (!symbol) {
    return NextResponse.json(
      { error: 'Symbol is required' },
      { status: 400 }
    );
  }
  // console.log(`Fetching quote for symbol: ${symbol}`);
  try {
    const quote = await getAlpacaLatestQuote(symbol);
    const price = quote.price;
    // console.log(`Quote for ${symbol}:`, quote);


    return NextResponse.json({
      symbol,
      price: price,
      timestamp: new Date().toISOString(),
      source: 'sip'
    });
  } catch (error) {
    console.error(`Error fetching quote for ${symbol}:`, error);
    return NextResponse.json(
      { error: 'Failed to fetch quote', details: error instanceof Error ? error.message : String(error) },
      { status: 500 }
    );
  }
}
