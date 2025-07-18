// components/pages/home/AccountOverview.tsx

"use client";
import AccountCards from "@/components/alpaca/AccountCards";
import { AccountGraph } from "@/components/alpaca/PortfolioHistory";

export function HomeLayout() {
  return (
    <div className="container mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6">Digital Transactions & Funds</h1>
      <h1 className="text-2xl font-bold mb-6">Dashboard</h1>
      <AccountCards />
      <AccountGraph />
    </div>
  );
}
