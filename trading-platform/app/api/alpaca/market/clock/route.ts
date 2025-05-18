import { NextResponse } from 'next/server';
// Use the dedicated function instead of trying to access the client directly
import { getAlpacaClock } from '@/lib/alpaca';

export async function GET() {
  try {
    // Use the dedicated function to get the clock
    const clock = await getAlpacaClock();
    
    return NextResponse.json({
      is_open: clock.is_open,
      next_open: clock.next_open,
      next_close: clock.next_close, 
      timestamp: clock.timestamp
    });
  } catch (error) {
    console.error('Error fetching market clock:', error);
    
    // Return a simple response to avoid potential serialization issues
    return NextResponse.json(
      { is_open: false }, // Default to closed on error
      { status: 200 }     // Return 200 to avoid breaking the UI
    );
  }
}
