'use client';
import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export function ServiceInspector() {
  const [url, setUrl] = useState('http://localhost:8001/health');
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const testEndpoint = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(url);
      const data = await response.json();
      setResult({
        status: response.status,
        statusText: response.statusText,
        data
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Service Inspector</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex gap-4 mb-4">
          <Input
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="Enter URL to test"
            className="flex-1"
          />
          <Button onClick={testEndpoint} disabled={loading}>
            {loading ? 'Testing...' : 'Test Endpoint'}
          </Button>
        </div>
        
        {error && (
          <div className="p-4 mb-4 bg-destructive/10 text-destructive rounded">
            Error: {error}
          </div>
        )}
        
        {result && (
          <div className="p-4 bg-muted rounded">
            <p>Status: {result.status} {result.statusText}</p>
            <pre className="mt-2 overflow-auto p-2 bg-background text-foreground rounded">
              {JSON.stringify(result.data, null, 2)}
            </pre>
          </div>
        )}
      </CardContent>
    </Card>
  );
} 