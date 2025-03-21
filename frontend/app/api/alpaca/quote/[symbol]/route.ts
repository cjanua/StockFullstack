import { NextRequest, NextResponse } from 'next/server';

export async function GET(
  request: NextRequest,
  { params }: { params: { symbol: string } }
) {
  const { symbol } = params;
  
  if (!symbol) {
    return NextResponse.json(
      { error: 'Symbol is required' },
      { status: 400 }
    );
  }
  
  try {
    // Since we're having issues with the Alpaca API for quotes, let's provide a fallback
    // This is a temporary solution until the actual quote API is working
    const mockPrice = 100 + (Math.random() * 50); // Random price between $100-$150
    
    return NextResponse.json({
      symbol,
      price: mockPrice,
      timestamp: new Date().toISOString(),
      source: 'mockData'
    });
  } catch (error) {
    console.error(`Error fetching quote for ${symbol}:`, error);
    return NextResponse.json(
      { error: 'Failed to fetch quote', details: error instanceof Error ? error.message : String(error) },
      { status: 500 }
    );
  }
}
