// app/api/auth/login/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { verifyUser, createSession } from '@/lib/db/sqlite';
import { cookies } from 'next/headers';

// Security configuration
const MAX_LOGIN_ATTEMPTS = 5;
const LOCKOUT_DURATION = 15 * 60 * 1000; // 15 minutes
const loginAttempts = new Map<string, { count: number; lastAttempt: Date }>();

// Whitelist configuration - ONLY user ID 1 is allowed
const WHITELISTED_USER_IDS = [1];
const AUDIT_MODE = false; // Enable strict security audit mode


// Rate limiting helper
function checkRateLimit(identifier: string): { allowed: boolean; remainingAttempts?: number } {
  const now = new Date();
  const attempts = loginAttempts.get(identifier);
  
  if (attempts) {
    const timeSinceLastAttempt = now.getTime() - attempts.lastAttempt.getTime();
    
    // Reset counter if lockout period has passed
    if (timeSinceLastAttempt > LOCKOUT_DURATION) {
      loginAttempts.delete(identifier);
      return { allowed: true };
    }
    
    // Check if user is locked out
    if (attempts.count >= MAX_LOGIN_ATTEMPTS) {
      return { allowed: false, remainingAttempts: 0 };
    }
    
    return { allowed: true, remainingAttempts: MAX_LOGIN_ATTEMPTS - attempts.count };
  }
  
  return { allowed: true, remainingAttempts: MAX_LOGIN_ATTEMPTS };
}

// Update login attempts
function recordLoginAttempt(identifier: string, success: boolean) {
  if (success) {
    loginAttempts.delete(identifier);
    return;
  }
  
  const attempts = loginAttempts.get(identifier) || { count: 0, lastAttempt: new Date() };
  attempts.count += 1;
  attempts.lastAttempt = new Date();
  loginAttempts.set(identifier, attempts);
}

// Security logging
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function logSecurityEvent(event: string, details: any) {
  const logEntry = {
    ...details,
    timestamp: new Date().toISOString(),
    auditMode: AUDIT_MODE,
  };
  
  // Use different log levels for different events
  switch (event) {
    case 'LOGIN_FAILED':
    case 'UNAUTHORIZED_USER_ATTEMPT':
    case 'LOGIN_RATE_LIMITED':
      console.error(`[SECURITY] ${new Date().toISOString()} - ${event}:`, logEntry);
      break;
    case 'LOGIN_SUCCESS':
      console.log(`[SECURITY] ${new Date().toISOString()} - ${event}:`, logEntry);
      break;
    default:
      console.warn(`[SECURITY] ${new Date().toISOString()} - ${event}:`, logEntry);
  }
}



export async function POST(request: NextRequest) {
  const clientIp = request.headers.get('x-forwarded-for') || 
                   request.headers.get('x-real-ip') || 
                   'unknown';

  try {
     // Parse request body
    let body;
    try {
      body = await request.json();
    } catch (parseError) {
      console.error('[LOGIN] Failed to parse request body:', parseError);
      return NextResponse.json(
        { error: 'Invalid request format' },
        { status: 400 }
      );
    }

    const rateLimitCheck = checkRateLimit(clientIp);
    if (!rateLimitCheck.allowed) {
      logSecurityEvent('LOGIN_RATE_LIMITED', { ip: clientIp });
      return NextResponse.json(
        { 
          error: 'Too many login attempts. Please try again later.',
          retryAfter: LOCKOUT_DURATION / 1000 // seconds
        },
        { status: 429 }
      );
    }

    const { username, password } = body;
    
    if (!username || !password) {
      return NextResponse.json(
        { error: 'Username and password are required' },
        { status: 400 }
      );
    }
    
    // Additional security checks
    if (username.length > 20 || password.length > 40) {
      logSecurityEvent('SUSPICIOUS_INPUT_LENGTH', { ip: clientIp, username });
      return NextResponse.json(
        { error: 'Invalid input' },
        { status: 400 }
      );
    }

    // Verify user credentials with timing attack protection
    const startTime = Date.now();
    const user = await verifyUser(username, password);
    
    // Add random delay to prevent timing attacks
    const processingTime = Date.now() - startTime;
    const minDelay = 100; // minimum 100ms
    const randomDelay = Math.random() * 200; // up to 200ms additional
    const totalDelay = Math.max(minDelay - processingTime, 0) + randomDelay;
    await new Promise(resolve => setTimeout(resolve, totalDelay));

    
    if (!user) {
      recordLoginAttempt(clientIp, false);
      logSecurityEvent('LOGIN_FAILED', { 
        ip: clientIp, 
        username,
        remainingAttempts: rateLimitCheck.remainingAttempts! - 1
      });

      return NextResponse.json(
        { error: 'Invalid username or password' },
        { status: 401 }
      );
    }

    // CRITICAL: Whitelist check - only allow user ID 1
    if (AUDIT_MODE && !WHITELISTED_USER_IDS.includes(user.id)) {
      logSecurityEvent('UNAUTHORIZED_USER_ATTEMPT', { 
        ip: clientIp, 
        userId: user.id,
        username: user.username,
        email: user.email
      });
      
      // Don't reveal that the user exists but is not whitelisted
      return NextResponse.json(
        { error: 'Invalid username or password' },
        { status: 401 }
      );
    }
    
    // Create session and get token
    const sessionId = await createSession(user.id, 24); // 24 hours

    if (!sessionId) {
      logSecurityEvent('SESSION_CREATION_FAILED', { ip: clientIp, userId: user.id });
      return NextResponse.json(
        { error: 'Failed to create session' },
        { status: 500 }
      );
    }
    // Record successful login
    recordLoginAttempt(clientIp, true);
    logSecurityEvent('LOGIN_SUCCESS', { 
      ip: clientIp, 
      userId: user.id,
      username: user.username,
      sessionId: sessionId.substring(0, 8) + '...' // Log partial session ID only
    });

    
    // Create response with user data
    const responseData = {
      success: true,
      user: {
        id: user.id,
        username: user.username,
        email: user.email,
        alpacaConnected: !!user.alpaca_key && user.alpaca_auth_status === 'active',
      },
    };
    
    // Create response
    const response = NextResponse.json(responseData);
    
    // Set token in cookie using the async cookies API
    const cookieStore = await cookies();
    cookieStore.set({
      name: 'auth_token',
      value: sessionId,
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'strict',
      path: '/',
      // 24 hours expiry
      maxAge: 60 * 60 * 24,
    });
    response.headers.set('X-Content-Type-Options', 'nosniff');
    response.headers.set('X-Frame-Options', 'DENY');
    response.headers.set('X-XSS-Protection', '1; mode=block');
    
    return response;
  } catch (error) {
    logSecurityEvent('LOGIN_ERROR', { 
      ip: clientIp, 
      error: error instanceof Error ? error.message : 'Unknown error' 
    });
    console.error('Login error:', error);
    return NextResponse.json(
      { error: 'An error occurred during login' },
      { status: 500 }
    );
  }
}