// middleware.ts
// import { NextResponse, type NextRequest } from "next/server";

// export function middleware(req: NextRequest) {
//   const alpacaToken = req.headers.get("session_token");

//   // Protected routes
//   if (req.nextUrl.pathname.startsWith("/dashboard")) {
//     if (!alpacaToken) {
//       return NextResponse.redirect(new URL("/api/auth/login", req.url));
//     }
//   }
//   return NextResponse.next();
// }

// export const config = {
//   matcher: ["/dashboard/:path*"],
// };

export function middleware() {
  return
}