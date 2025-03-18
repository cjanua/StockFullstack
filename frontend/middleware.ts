// middleware.ts
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

// Paths that don't require authentication
const publicPaths = ['/login', '/register', '/api/auth/login', '/api/auth/register'];

export async function middleware(request: NextRequest) {
  const path = request.nextUrl.pathname;
  
  // Allow access to public paths
  if (publicPaths.some(publicPath => path.startsWith(publicPath))) {
    return NextResponse.next();
  }
  
  // Check for authentication token in cookies
  // Note: In middleware, we can't use the async cookies API, so we use the request.cookies directly
  const authToken = request.cookies.get('auth_token')?.value;
  
  if (!authToken) {
    // Redirect to login page if not authenticated
    const url = new URL('/login', request.url);
    url.searchParams.set('redirect', encodeURIComponent(request.url));
    return NextResponse.redirect(url);
  }
  
  // Continue to the protected route
  return NextResponse.next();
}

// Configure middleware to run on specific paths
export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     * - public folder
     */
    '/((?!_next/static|_next/image|favicon.ico|public/).*)',
  ],
};