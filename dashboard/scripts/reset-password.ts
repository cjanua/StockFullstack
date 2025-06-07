// scripts/reset-password.ts
// Run with: npx tsx scripts/reset-password.ts

import bcrypt from 'bcryptjs';
import sqlite3 from 'sqlite3';
import { open } from 'sqlite';
import { join } from 'path';

async function resetPassword(username: string, newPassword: string) {
  const dbPath = join(process.cwd(), 'data', 'auth.db');
  
  const db = await open({
    filename: dbPath,
    driver: sqlite3.Database,
  });
  
  // Hash the new password
  const passwordHash = await bcrypt.hash(newPassword, 10);
  
  // Update the password
  const result = await db.run(
    'UPDATE users SET password_hash = ? WHERE username = ?',
    [passwordHash, username]
  );
  
  if (result.changes === 0) {
    console.error('User not found');
  } else {
    console.log(`Password reset successfully for user: ${username}`);
  }
  
  await db.close();
}

// Get command line arguments
const username = process.argv[2];
const newPassword = process.argv[3];

if (!username || !newPassword) {
  console.log('Usage: npx tsx scripts/reset-password.ts <username> <new-password>');
  process.exit(1);
}

if (newPassword.length < 8) {
  console.error('Password must be at least 8 characters long');
  process.exit(1);
}

resetPassword(username, newPassword).catch(console.error);