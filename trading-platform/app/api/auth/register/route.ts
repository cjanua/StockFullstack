// app/api/auth/register/route.ts
import { NextRequest, NextResponse } from 'next/server';

// REGISTRATION IS TEMPORARILY DISABLED FOR SECURITY AUDIT
const REGISTRATION_ENABLED = false;

export async function POST(request: NextRequest) {
  // Log registration attempt for security monitoring
  const clientIp = request.headers.get('x-forwarded-for') || 
                   request.headers.get('x-real-ip') || 
                   'unknown';
  
  console.log(`[SECURITY] ${new Date().toISOString()} - REGISTRATION_ATTEMPT_BLOCKED:`, {
    ip: clientIp,
    timestamp: new Date().toISOString(),
    registrationEnabled: REGISTRATION_ENABLED,
  });
  
  // Return 403 Forbidden with security message
  return NextResponse.json(
    { 
      error: 'Registration is temporarily disabled for security maintenance.',
      message: 'Please contact your system administrator for account creation.'
    },
    { status: 403 }
  );
}

// Also block other HTTP methods for extra security
export async function GET() {
  return NextResponse.json({ error: 'Method not allowed' }, { status: 405 });
}

export async function PUT() {
  return NextResponse.json({ error: 'Method not allowed' }, { status: 405 });
}

export async function DELETE() {
  return NextResponse.json({ error: 'Method not allowed' }, { status: 405 });
}
// export async function POST(request: NextRequest) {
//   try {
//     const body = await request.json();
//     const { username, email, password } = body;
    
//     // Basic validation
//     if (!username || !email || !password) {
//       return NextResponse.json(
//         { error: 'Username, email, and password are required' },
//         { status: 400 }
//       );
//     }
    
//     // Validate email format
//     const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
//     if (!emailRegex.test(email)) {
//       return NextResponse.json(
//         { error: 'Invalid email format' },
//         { status: 400 }
//       );
//     }
    
//     // Validate password strength
//     if (password.length < 8) {
//       return NextResponse.json(
//         { error: 'Password must be at least 8 characters long' },
//         { status: 400 }
//       );
//     }
    
//     // Create user
//     const user = await createUser(username, email, password);
    
//     if (!user) {
//       return NextResponse.json(
//         { error: 'Failed to create user. Username or email may already exist.' },
//         { status: 400 }
//       );
//     }
    
//     // Return success response
//     return NextResponse.json({
//       success: true,
//       user: {
//         id: user.id,
//         username: user.username,
//         email: user.email,
//       },
//     });
//   } catch (error) {
//     console.error('Registration error:', error);
//     return NextResponse.json(
//       { error: 'An error occurred during registration' },
//       { status: 500 }
//     );
//   }
// }