'use client';
import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export function DirectApiTest() {
  const [result, setResult] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const testApi = async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Try direct request to FastAPI - no Next.js API proxy
      const url = 'http://localhost:8001/api/portfolio/recommendations';
      console.log(`Testing direct API call to: ${url}`);
      
      const response = await fetch(url, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      });
      
      console.log(`Response status: ${response.status} ${response.statusText}`);
      
      if (!response.ok) {
        throw new Error(`API error: ${response.status} ${response.statusText}`);
      }
      
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Direct API Test</CardTitle>
      </CardHeader>
      <CardContent>
        <Button onClick={testApi} disabled={loading} className="mb-4">
          {loading ? 'Testing...' : 'Test Direct API Call'}
        </Button>
        
        {error && (
          <div className="p-4 mb-4 bg-destructive/10 text-destructive rounded">
            Error: {error}
          </div>
        )}
        
        {result && (
          <div className="p-4 bg-muted rounded">
            <pre className="overflow-auto p-2 bg-background text-foreground rounded">
              {JSON.stringify(result, null, 2)}
            </pre>
          </div>
        )}
      </CardContent>
    </Card>
  );
} 