// app/api/auth/logout/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { deleteSession } from '@/lib/db/sqlite';

export async function POST(request: NextRequest) {
  try {
    // Get auth token from cookies using the async cookies API
    const cookieStore = await cookies();
    const authToken = cookieStore.get('auth_token')?.value;
    
    if (authToken) {
      // Delete session from database
      await deleteSession(authToken);
      
      // Create response
      const response = NextResponse.json({ success: true });
      
      // Clear the auth cookie
      cookieStore.delete('auth_token');
      
      return response;
    }
    
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Logout error:', error);
    return NextResponse.json(
      { error: 'An error occurred during logout' },
      { status: 500 }
    );
  }
}