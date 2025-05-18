import { NextRequest, NextResponse } from "next/server";

export async function GET(req: NextRequest) {
  // Get query parameters from the URL
  const searchParams = req.nextUrl.searchParams;
  
  const lookbackDays = searchParams.get('lookback_days') || '365';
  const minChangePercent = searchParams.get('min_change_percent') || '0.01';
  const cashReservePercent = searchParams.get('cash_reserve_percent') || '0.05';
  
  try {
    // Health check is already confirmed working via curl test
    
    // IMPORTANT: Use exactly the same endpoint path that's working in curl
    const serviceUrl = `http://localhost:8001/api/portfolio/recommendations?lookback_days=${lookbackDays}&min_change_percent=${minChangePercent}&cash_reserve_percent=${cashReservePercent}`;
    console.log(`Making request to: ${serviceUrl}`);
    
    const response = await fetch(serviceUrl, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      cache: 'no-store',
    });
    
    console.log(`Response status: ${response.status} ${response.statusText}`);
    
    if (!response.ok) {
      let errorMessage = 'Failed to get recommendations';
      try {
        const errorData = await response.json();
        console.log("Error data:", errorData);
        errorMessage = errorData.detail || errorMessage;
      } catch (e) {
        // If parsing JSON fails, use the status text
        console.error("Failed to parse error response:", e);
        errorMessage = response.statusText || errorMessage;
      }
      
      return NextResponse.json(
        { error: errorMessage }, 
        { status: response.status }
      );
    }
    
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error fetching portfolio recommendations:', error);
    return NextResponse.json(
      { error: 'Failed to fetch portfolio recommendations: ' + (error instanceof Error ? error.message : String(error)) }, 
      { status: 500 }
    );
  }
} 