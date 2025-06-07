// app/api/alpaca/connect/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { getUserBySessionToken } from '@/lib/db/sqlite';
import { connectUserToAlpaca } from '@/lib/services/alpacaAuth';

export async function POST(request: NextRequest) {
  try {
    // Get auth token from cookies using the async cookies API
    const cookieStore = await cookies();
    const authToken = cookieStore.get('auth_token')?.value;
    
    if (!authToken) {
      return NextResponse.json(
        { error: 'Not authenticated' },
        { status: 401 }
      );
    }
    
    // Get user by session token
    const user = await getUserBySessionToken(authToken);
    
    if (!user) {
      return NextResponse.json(
        { error: 'Invalid or expired session' },
        { status: 401 }
      );
    }
    
    // Get Alpaca credentials from request body
    const body = await request.json();
    const { alpaca_key, alpaca_secret, paper } = body;
    
    if (!alpaca_key || !alpaca_secret) {
      return NextResponse.json(
        { error: 'Alpaca API key and secret are required' },
        { status: 400 }
      );
    }
    
    // Connect user to Alpaca
    const result = await connectUserToAlpaca(user.id, alpaca_key, alpaca_secret, paper);
    
    if (!result.success) {
      return NextResponse.json(
        { error: result.message },
        { status: 400 }
      );
    }
    
    return NextResponse.json({
      success: true,
      message: result.message,
    });
  } catch (error) {
    console.error('Alpaca connect error:', error);
    return NextResponse.json(
      { error: 'An error occurred connecting to Alpaca' },
      { status: 500 }
    );
  }
}