import crypto from 'crypto';

// Generate a strong token (do this once and save the value)
export function generateSecureToken(length = 64) {
  return crypto.randomBytes(length).toString('hex');
}

// Store your generated token in an environment variable
// This should match between your frontend and backend
const ACCESS_TOKEN = process.env.ACCESS_TOKEN || 'your-pre-generated-secure-token';

// Verify the token
export function verifyToken(token: string): boolean {
  // Use a constant-time comparison to prevent timing attacks
  return crypto.timingSafeEqual(
    Buffer.from(token),
    Buffer.from(ACCESS_TOKEN)
  );
}