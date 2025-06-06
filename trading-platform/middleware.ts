
// middleware.ts
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

// List of public paths that don't require authentication
export const publicPaths = ['/login', '/register', '/api/auth/login', '/api/auth/register'];

// Security headers to apply to all responses
const securityHeaders = {
  'X-Content-Type-Options': 'nosniff',
  'X-Frame-Options': 'DENY',
  'X-XSS-Protection': '1; mode=block',
  'Referrer-Policy': 'strict-origin-when-cross-origin',
  'Permissions-Policy': 'camera=(), microphone=(), geolocation=()',
  'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
  'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdnjs.cloudflare.com; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' data:; connect-src 'self' https://data.alpaca.markets wss://stream.alpaca.markets http://localhost:8001; frame-src 'none'; object-src 'none'; base-uri 'self'; form-action 'self';"
};

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;
  
  // Security: Strip any query parameters from login requests to prevent password leakage
  if (pathname === '/login' && request.nextUrl.search) {
    const url = new URL(request.url);
    const redirect = url.searchParams.get('redirect');
    
    // Only preserve the redirect parameter, remove everything else
    url.search = '';
    if (redirect) {
      url.searchParams.set('redirect', redirect);
    }
    
    // Log security warning if password was in URL
    if (url.searchParams.get('password')) {
      console.error('[SECURITY WARNING] Password detected in URL parameters!');
    }
    
    return NextResponse.redirect(url);
  }
  
  // Block access to register page completely
  if (pathname === '/register' || pathname.startsWith('/api/auth/register')) {
    console.log(`[SECURITY] Registration attempt blocked: ${pathname}`);
    return NextResponse.redirect(new URL('/login', request.url));
  }
  if (pathname.startsWith('/api/')) {
    const response = NextResponse.next();
    
    // Apply security headers to API responses
    Object.entries(securityHeaders).forEach(([key, value]) => {
      response.headers.set(key, value);
    });
    
    // Add request ID for tracking
    const requestId = crypto.randomUUID();
    response.headers.set('X-Request-ID', requestId);
    
    // Log API requests
    console.log(`[API] ${new Date().toISOString()} - ${request.method} ${pathname} - Request ID: ${requestId}`);
    
    return response;
  }

  // Check if the path is public
  const isPublicPath = publicPaths.some(path => pathname.startsWith(path));
  
  // Get the auth token from cookies
  const authToken = request.cookies.get('auth_token')?.value;
  
  // Create response
  let response: NextResponse;
  
  // Redirect logic
  if (!authToken && !isPublicPath) {
    // Not authenticated and trying to access protected route
    const loginUrl = new URL('/login', request.url);
    loginUrl.searchParams.set('redirect', pathname);
    response = NextResponse.redirect(loginUrl);
  } else if (authToken && pathname === '/login') {
    // Already authenticated and trying to access login
    response = NextResponse.redirect(new URL('/account', request.url));
  } else {
    // Allow the request to continue
    response = NextResponse.next();
  }
  
  // Apply security headers to all responses
  Object.entries(securityHeaders).forEach(([key, value]) => {
    response.headers.set(key, value);
  });
  
  // Add request ID for tracking
  const requestId = crypto.randomUUID();
  response.headers.set('X-Request-ID', requestId);
  
  return response;
}

// Configure which routes the middleware should run on
// Using the new Next.js 14 matcher configuration
export const config = {
  matcher: [
    /*
     * Match all request paths except:
     * - _next/static (static files)
     * - _next/image (image optimization files)  
     * - favicon.ico (favicon file)
     * - public folder
     * - .well-known (for SSL/security checks)
     */
    '/((?!_next/static|_next/image|favicon.ico|public/|.well-known).*)',
  ],
};