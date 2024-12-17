// app/api/auth/logout/route.ts
import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import prisma from "@/lib/userdb";

export async function POST() {
  const cookieStore = await cookies();
  const sessionToken = cookieStore.get("session_token")?.value;

  if (sessionToken) {
    await prisma.user.update({
      where: { sessionToken },
      data: { sessionToken: null },
    });

    cookieStore.delete("session_token");
  }

  redirect("/");
}
