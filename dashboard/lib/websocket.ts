/* eslint-disable @typescript-eslint/no-explicit-any */
// dashboard/lib/websocket.ts
"use client";

import { create } from "zustand";
import React from "react";

// Define the allowed WebSocket message types
export type TradeUpdate = {
  type: "trade_update";
  data: any;
};

export type QuoteUpdate = {
  type: "quote";
  data: any;
};

export type PositionUpdate = {
  type: "position_update";
  data: any;
};

export type WebSocketMessage = TradeUpdate | QuoteUpdate | PositionUpdate;

// Create the WebSocket store
interface WebSocketStore {
  socket: WebSocket | null;
  connected: boolean;
  messages: WebSocketMessage[];
  error: string | null;
  subscribedSymbols: string[];
  
  connect: () => void;
  disconnect: () => void;
  subscribe: (symbols: string[]) => void;
  unsubscribe: (symbols: string[]) => void;
  clearMessages: () => void;
}

export const useWebSocketStore = create<WebSocketStore>((set, get) => ({
  socket: null,
  connected: false,
  messages: [],
  error: null,
  subscribedSymbols: [],
  
  connect: () => {
    // Close any existing connection
    if (get().socket) {
      get().socket?.close();
    }
    
    try {
      // Create WebSocket connection
      // Use environment variable or default to test API
      const apiKey = process.env.NEXT_PUBLIC_ALPACA_KEY || "";
      const apiSecret = process.env.NEXT_PUBLIC_ALPACA_SECRET || "";
      
      // Using basic auth format: apiKey:apiSecret
      // const authInfo = btoa(`${apiKey}:${apiSecret}`);
      
      const socket = new WebSocket("wss://stream.alpaca.markets/v2/iex");
      
      socket.onopen = () => {
        set({ connected: true, error: null });
        // Authenticate
        socket.send(JSON.stringify({
          action: "auth",
          key: apiKey,
          secret: apiSecret
        }));
        
        // Resubscribe to previous symbols if any
        const { subscribedSymbols } = get();
        if (subscribedSymbols.length > 0) {
          socket.send(JSON.stringify({
            action: "subscribe",
            trades: subscribedSymbols,
            quotes: subscribedSymbols
          }));
        }
        
        // Also subscribe to trade updates (account activity)
        socket.send(JSON.stringify({
          action: "listen",
          data: {
            streams: ["trade_updates"]
          }
        }));
      };
      
      socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          // Handle authentication response
          if (data.stream === "authorization") {
            if (data.data.status === "authorized") {
              console.log("WebSocket authorized");
            } else {
              set({ error: "WebSocket authorization failed" });
            }
            return;
          }
          
          // Handle trade updates (order fills, etc.)
          if (data.stream === "trade_updates") {
            const newMessage: TradeUpdate = { 
              type: "trade_update", 
              data: data.data 
            };
            
            set((state) => ({
              messages: [newMessage, ...state.messages].slice(0, 100)
            }));
            return;
          }
          
          // Handle quotes
          if (data.stream === "quotes") {
            const newMessage: QuoteUpdate = { 
              type: "quote", 
              data: data.data 
            };
            
            set((state) => ({
              messages: [newMessage, ...state.messages].slice(0, 100)
            }));
            return;
          }
          
        } catch (error) {
          console.error("Error parsing WebSocket message", error);
        }
      };
      
      socket.onerror = (error) => {
        console.error("WebSocket error", error);
        set({ error: "WebSocket connection error" });
      };
      
      socket.onclose = () => {
        set({ connected: false });
      };
      
      set({ socket });
    } catch (error) {
      console.error("Failed to connect WebSocket", error);
      set({ error: "Failed to connect to WebSocket" });
    }
  },
  
  disconnect: () => {
    get().socket?.close();
    set({ socket: null, connected: false });
  },
  
  subscribe: (symbols: string[]) => {
    const { socket, connected, subscribedSymbols } = get();
    
    // Add new symbols without duplicates
    const uniqueNewSymbols = symbols.filter(
      (symbol) => !subscribedSymbols.includes(symbol)
    );
    
    if (uniqueNewSymbols.length === 0) return;
    
    const updatedSymbols = [...subscribedSymbols, ...uniqueNewSymbols];
    
    if (connected && socket) {
      socket.send(JSON.stringify({
        action: "subscribe",
        trades: uniqueNewSymbols,
        quotes: uniqueNewSymbols
      }));
    }
    
    set({ subscribedSymbols: updatedSymbols });
  },
  
  unsubscribe: (symbols: string[]) => {
    const { socket, connected, subscribedSymbols } = get();
    
    const updatedSymbols = subscribedSymbols.filter(
      (symbol) => !symbols.includes(symbol)
    );
    
    if (connected && socket) {
      socket.send(JSON.stringify({
        action: "unsubscribe",
        trades: symbols,
        quotes: symbols
      }));
    }
    
    set({ subscribedSymbols: updatedSymbols });
  },
  
  clearMessages: () => {
    set({ messages: [] });
  }
}));

// Helper hook to auto-connect WebSocket when component mounts
export function useWebSocket(symbols?: string[]) {
  const { 
    connect, disconnect, subscribe, 
    connected, messages, error 
  } = useWebSocketStore();
  
  React.useEffect(() => {
    // Connect on mount
    connect();
    
    // Subscribe to symbols if provided
    if (symbols && symbols.length > 0) {
      subscribe(symbols);
    }
    
    // Disconnect on unmount
    return () => {
      disconnect();
    };
  }, [connect, disconnect, subscribe, symbols]);
  
  // Subscribe to new symbols when they change
  React.useEffect(() => {
    if (symbols && symbols.length > 0) {
      subscribe(symbols);
    }
  }, [symbols, subscribe]);
  
  return { connected, messages, error };
}