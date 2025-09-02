// dashboard/app/api/debug/portfolio-service/route.ts

import { NextResponse } from "next/server";

export async function GET() {
  const endpoints = [
    "http://localhost:8001/health",
    "http://localhost:8001/recommendations",
    "http://localhost:8001/optimize",
  ];
  
  const results: Record<string, object> = {};
  
  for (const endpoint of endpoints) {
    try {
      console.log(`Testing endpoint: ${endpoint}`);
      const response = await fetch(endpoint, {
        method: 'GET',
        cache: 'no-store',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      results[endpoint] = {
        status: response.status,
        statusText: response.statusText,
        ok: response.ok,
        body: response.status === 200 ? await response.json() : null
      };
    } catch (error) {
      results[endpoint] = {
        error: true,
        message: error instanceof Error ? error.message : String(error)
      };
    }
  }
  
  return NextResponse.json(results);
} 