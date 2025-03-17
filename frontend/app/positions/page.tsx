"use client";
import { PositionTable } from "@/components/alpaca/PositionTable";
import { OrderForm } from "@/components/alpaca/OrderForm";
// import { PositionLiveUpdates } from "@/components/alpaca/PositionLiveUpdates";

export default function PositionsLayout() {
  return (
    <div className="p-6">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold">Open Positions</h1>
        <OrderForm />
      </div>
      
      {/* <PositionLiveUpdates /> */}
      
      <PositionTable count={15} />
    </div>
  );
}