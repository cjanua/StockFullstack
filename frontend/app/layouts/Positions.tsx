"use client";
import { PositionTable } from "@/components/pages/positions/PositionTable";

export function PositionsLayout() {
  return (
    <div className="container mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6">Open Positions</h1>
      <PositionTable count={15} />
    </div>
  );
}
