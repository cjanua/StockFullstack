// // app/auth/success/page.tsx
// "use client";

// import { useEffect } from "react";
// import { useRouter, useSearchParams } from "next/navigation";
// import {  } from "@/lib/auth";

// export default function AuthSuccess() {
//   const router = useRouter();
//   const searchParams = useSearchParams();
//   const token = searchParams.get("token");

//   useEffect(() => {
//     if (token) {
//       setAlpacaToken(token);
//       router.push("/");
//     }
//   }, [token, router]);

//   return <div>Authenticating...</div>;
// }
