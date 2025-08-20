// dashboard/app/api/auth/me/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { getUserBySessionToken } from '@/lib/db/sqlite';
import { checkAlpacaConnection } from '@/lib/services/alpacaAuth';

export async function GET(_request: NextRequest): Promise<NextResponse> {
  try {
    // Get auth token from cookies using the async cookies API
    const cookieStore = await cookies();
    const authToken = cookieStore.get('auth_token')?.value;

    if (!authToken) {
      return NextResponse.json({ error: 'Not authenticated' }, { status: 401 });
    }

    // Get user by session token
    const user = await getUserBySessionToken(authToken);
    if (!user) {
      // Clear invalid token
      cookieStore.delete('auth_token');
      return NextResponse.json({ error: 'Invalid or expired session' }, { status: 401 });
    }

    // Check Alpaca connection status if credentials exist
    let alpacaStatus = { 
      status: 'inactive' as 'active' | 'inactive' | 'error', 
      message: 'No Alpaca credentials have been set.'
    };
    if (user.alpaca_key && user.alpaca_secret) {
      alpacaStatus = await checkAlpacaConnection(user);
    }

    // Return user info (without sensitive data)
    return NextResponse.json({
      id: user.id,
      username: user.username || 'Unknown',
      email: user.email || 'Unknown',
      created_at: user.created_at || new Date().toISOString(),
      alpaca: {
        connected: alpacaStatus.status === 'active',
        status: alpacaStatus.status,
        message: alpacaStatus.message,
        has_credentials: !!user.alpaca_key && !!user.alpaca_secret,
        last_verified: user.last_verified || null,
        use_paper_trading: user.use_paper_trading === 1, // Convert INTEGER to boolean
      },
    });
  } catch (error) {
    console.error('Auth check error:', error);
    return NextResponse.json(
      { error: 'An error occurred checking authentication' },
      { status: 500 }
    );
  }
}