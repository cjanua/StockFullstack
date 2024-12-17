// types/auth.ts
export interface AlpacaUser {
  id: string;
  alpacaId: string;
  email: string | null; // Make nullable to match schema
  password: string | null; // Make nullable to match schema
  isEmailVerified: boolean;
  accessToken: string;
  sessionToken: string | null; // Add sessionToken from schema
  createdAt: Date; // Add createdAt from schema
  updatedAt: Date; // Add updatedAt from schema
}
