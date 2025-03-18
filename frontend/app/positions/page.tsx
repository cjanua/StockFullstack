// app/positions/page.tsx
"use client";
import { PositionTable } from "@/components/alpaca/PositionTable";
import { OrderForm } from "@/components/alpaca/OrderForm";
import { PortfolioRecommendations } from "@/components/alpaca/PortfolioRecommendations";

export default function PositionsPage() {
  return (
    <div className="p-6">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold">Open Positions</h1>
        <OrderForm />
      </div>
      
      <div className="mb-8">
        <PortfolioRecommendations />
      </div>
      
      <PositionTable count={15} />
    </div>
  );
}