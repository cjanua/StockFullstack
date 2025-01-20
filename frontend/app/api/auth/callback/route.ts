// // app/api/auth/callback/route.ts
// import { Account, getAlpacaAccount } from "@/lib/alpaca";
// import { ALPACA_OAUTH_CONFIG } from "@/lib/config";
// import { cookies } from "next/headers";
// import { redirect } from "next/navigation";
// import { NextRequest, NextResponse } from "next/server";

// export async function GET(request: NextRequest) {
//   const searchParams = request.nextUrl.searchParams;
//   const code = searchParams.get("code");
//   const state = searchParams.get("state");

//   if (!code) {
//     return NextResponse.redirect("/auth/error");
//   }

//   const cookieStore = await cookies();
//   const storedState = cookieStore.get("auth_state")?.value;

//   if (!code || !state || state !== storedState) {
//     throw new Error("Invalid state or code");
//   }

//   try {
//     console.log("Client ID:", process.env.ALPACA_CLIENT_ID);
//     console.log("Redirect URI:", ALPACA_OAUTH_CONFIG.redirect_uri);
//     const tokenResponse = await fetch(ALPACA_OAUTH_CONFIG.token_url, {
//       method: "POST",
//       headers: {
//         "Content-Type": "application/x-www-form-urlencoded",
//       },
//       body: new URLSearchParams({
//         grant_type: "authorization_code",
//         code,
//         client_id: process.env.ALPACA_CLIENT_ID!,
//         client_secret: process.env.ALPACA_CLIENT_SECRET!,
//         redirect_uri: ALPACA_OAUTH_CONFIG.redirect_uri,
//       }),
//     });

//     const { access_token } = await tokenResponse.json();

//     console.log("Token exchange success:", access_token);

//     // Get Alpaca account
//     const account: Account = await getAlpacaAccount(access_token);

//     const sessionToken = crypto.randomUUID();
//     const user = await prisma.user.upsert({
//       where: { alpacaId: account.id },
//       update: {
//         accessToken: access_token,
//         sessionToken,
//       },
//       create: {
//         alpacaId: account.id,
//         accessToken: access_token,
//         sessionToken,
//         isEmailVerified: false,
//       },
//     });

//     // Set session cookie
//     cookieStore.set("session_token", sessionToken, {
//       httpOnly: true,
//       secure: process.env.NODE_ENV === "production",
//       sameSite: "lax",
//     });

//     // Redirect based on registration status
//     if (!user.email || !user.password) {
//       redirect("/complete-registration");
//     }

//     redirect("/dashboard");
//   } catch (error) {
//     console.error("Token exchange error:", error);
//     return NextResponse.redirect("/auth/error");
//   }
// }
