// app/page.tsx
import { AccountOverview } from "@/app/layouts/Home";
import { PortfolioGraph } from "@/components/plotting/Portfolio";
import { verifySession } from "@/lib/auth";
import { redirect } from "next/navigation";

export default async function Home() {
  // const user = await verifySession();

  // if (!user.email || !user.password) {
  //   redirect("/register");
  // }

  return <>
    <AccountOverview />
    {/* <PortfolioGraph /> */}
  </>;
}
