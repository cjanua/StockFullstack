// frontend/components/alpaca/PositionLiveUpdates.tsx
"use client";

import { useEffect, useState } from "react";
import { Position } from "@/lib/alpaca";
import { useWebSocket } from "@/lib/websocket";
import { AlertCircle, ArrowDown, ArrowUp, Wifi, WifiOff } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { usePositions } from "@/hooks/alpaca/usePositions";
import { fmtCurrency, fmtPercent } from "@/lib/utils";

export function PositionLiveUpdates() {
  const { positions } = usePositions();
  const [symbols, setSymbols] = useState<string[]>([]);
  const [updates, setUpdates] = useState<Map<string, { price: number, change: number }>>(new Map());
  
  // Extract symbols from positions
  useEffect(() => {
    if (positions && positions.length > 0) {
      const positionSymbols = positions.map(p => p.symbol);
      setSymbols(positionSymbols);
    }
  }, [positions]);
  
  // Connect to WebSocket with our position symbols
  const { connected, messages } = useWebSocket(symbols);
  
  // Process incoming WebSocket messages
  useEffect(() => {
    if (messages && messages.length > 0) {
      const latestMessage = messages[0];
      
      if (latestMessage.type === "quote") {
        const { symbol, bidprice, askprice } = latestMessage.data;
        const midPrice = (bidprice + askprice) / 2;
        
        // Find the position for this symbol to calculate change
        const position = positions?.find(p => p.symbol === symbol);
        if (position) {
          const oldPrice = parseFloat(position.current_price);
          const change = midPrice - oldPrice;
          
          setUpdates(prev => {
            const newMap = new Map(prev);
            newMap.set(symbol, { price: midPrice, change });
            return newMap;
          });
        }
      }
    }
  }, [messages, positions]);
  
  if (!positions || positions.length === 0) {
    return null;
  }
  
  return (
    <div className="mb-4">
      <div className="flex items-center gap-2 mb-2">
        <h3 className="text-sm font-medium">Live Updates</h3>
        {connected ? (
          <Badge variant="outline" className="bg-green-950 text-green-500">
            <Wifi className="h-3 w-3 mr-1" /> Connected
          </Badge>
        ) : (
          <Badge variant="outline" className="bg-red-950 text-red-500">
            <WifiOff className="h-3 w-3 mr-1" /> Disconnected
          </Badge>
        )}
      </div>
      
      {updates.size > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-2">
          {Array.from(updates.entries()).map(([symbol, data]) => (
            <div 
              key={symbol}
              className="flex items-center p-2 rounded-md border border-border"
            >
              <div className="font-medium mr-2">{symbol}</div>
              <div className={data.change > 0 ? "text-green-500" : "text-red-500"}>
                {fmtCurrency(data.price)} 
                <span className="ml-1 text-xs">
                  {data.change > 0 ? <ArrowUp className="h-3 w-3 inline" /> : <ArrowDown className="h-3 w-3 inline" />}
                  {fmtPercent(Math.abs(data.change / (data.price - data.change)))}
                </span>
              </div>
            </div>
          ))}
        </div>
      ) : connected ? (
        <div className="text-sm text-muted-foreground">Waiting for price updates...</div>
      ) : (
        <div className="flex items-center text-sm text-amber-500">
          <AlertCircle className="h-4 w-4 mr-1" />
          Not receiving live updates. Please check your connection.
        </div>
      )}
    </div>
  );
}