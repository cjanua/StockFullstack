// components/pages/home/AccountOverview.tsx

"use client";
import AccountCards from "@/components/pages/home/AccountCards";
import { AccountGraph } from "@/components/plots/Account";
import { useAccount } from "@/hooks/alpaca/useAccount";

export function HomeLayout() {
  return (
    <div className="container mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6">Financial Dashboard</h1>
      <AccountCards />
      <AccountGraph />
    </div>
  );
}
