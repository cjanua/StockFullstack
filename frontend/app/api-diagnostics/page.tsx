"use client";
import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Loader2, CheckCircle, XCircle, AlertTriangle, Activity } from 'lucide-react';
import { ServiceInspector } from '@/components/ServiceInspector';
import { DirectApiTest } from '@/components/DirectApiTest';

// Define the type for your state
interface CheckStatus {
  status: string;
  data: null;
  error: string | null; // Allow error to be a string or null
}

interface ChecksState {
  health: CheckStatus;
  root: CheckStatus;
  recommendations: CheckStatus;
}

export default function ApiDiagnosticsPage() {
  const [checks, setChecks] = useState<ChecksState>({
    health: { status: 'pending', data: null, error: null },
    root: { status: 'pending', data: null, error: null },
    recommendations: { status: 'pending', data: null, error: null }
  });
  
  const [isRunningChecks, setIsRunningChecks] = useState(false);
  
  const runChecks = async () => {
    setIsRunningChecks(true);
    setChecks({
      health: { status: 'loading', data: null, error: null },
      root: { status: 'loading', data: null, error: null },
      recommendations: { status: 'loading', data: null, error: null }
    });
    
    // Test health endpoint
    try {
      const healthResponse = await fetch('http://localhost:8001/health');
      const healthData = await healthResponse.json();
      setChecks(prev => ({
        ...prev,
        health: { 
          status: healthResponse.ok ? 'success' : 'error', 
          data: healthData, 
          error: healthResponse.ok ? null : `HTTP ${healthResponse.status} ${healthResponse.statusText}` 
        }
      }));
    } catch (error) {
      setChecks(prev => ({
        ...prev,
        health: { 
          status: 'error', 
          data: null, 
          error: error instanceof Error ? error.message : String(error) 
        }
      }));
    }
    
    // Test root endpoint
    try {
      const rootResponse = await fetch('http://localhost:8001/');
      const rootData = await rootResponse.json();
      setChecks(prev => ({
        ...prev,
        root: { 
          status: rootResponse.ok ? 'success' : 'error', 
          data: rootData, 
          error: rootResponse.ok ? null : `HTTP ${rootResponse.status} ${rootResponse.statusText}` 
        }
      }));
    } catch (error) {
      setChecks(prev => ({
        ...prev,
        root: { 
          status: 'error', 
          data: null, 
          error: error instanceof Error ? error.message : String(error) 
        }
      }));
    }
    
    // Test direct recommendations endpoint
    try {
      const recommendationsResponse = await fetch('http://localhost:8001/api/portfolio/recommendations');
      let recommendationsData = null;
      
      try {
        recommendationsData = await recommendationsResponse.json();
      } catch (e) {
        // If we can't parse JSON, capture the text
        recommendationsData = await recommendationsResponse.text();
      }
      
      setChecks(prev => ({
        ...prev,
        recommendations: { 
          status: recommendationsResponse.ok ? 'success' : 'error', 
          data: recommendationsData, 
          error: recommendationsResponse.ok ? null : `HTTP ${recommendationsResponse.status} ${recommendationsResponse.statusText}` 
        }
      }));
    } catch (error) {
      setChecks(prev => ({
        ...prev,
        recommendations: { 
          status: 'error', 
          data: null, 
          error: error instanceof Error ? error.message : String(error) 
        }
      }));
    }
    
    setIsRunningChecks(false);
  };
  
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'success': 
        return <CheckCircle className="h-5 w-5 text-green-500" />;
      case 'error': 
        return <XCircle className="h-5 w-5 text-red-500" />;
      case 'loading': 
        return <Loader2 className="h-5 w-5 animate-spin text-blue-500" />;
      default: 
        return <AlertTriangle className="h-5 w-5 text-gray-400" />;
    }
  };
  
  const renderResult = (check: CheckStatus) => {
    if (check.status === 'pending') {
      return <div className="text-muted-foreground">Not tested yet</div>;
    }
    
    if (check.status === 'loading') {
      return <div className="text-blue-500">Testing...</div>;
    }
    
    if (check.status === 'error') {
      return (
        <div className="p-3 bg-destructive/10 border border-destructive/20 rounded text-destructive">
          <p className="font-semibold">Error: {check.error}</p>
        </div>
      );
    }
    
    return (
      <div className="p-3 bg-green-500/10 dark:bg-green-500/20 border border-green-500/20 rounded">
        <p className="text-green-700 dark:text-green-400 font-semibold">Success!</p>
        <pre className="mt-2 p-2 bg-background border text-sm overflow-auto max-h-48 rounded">
          {typeof check.data === 'string' ? check.data : JSON.stringify(check.data, null, 2)}
        </pre>
      </div>
    );
  };
  
  return (
    <div className="container mx-auto py-8 space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold flex items-center">
          <Activity className="mr-2 h-6 w-6" />
          API Diagnostics Dashboard
        </h1>
        <Button 
          onClick={runChecks} 
          disabled={isRunningChecks}
          size="lg"
        >
          {isRunningChecks ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Running Tests...
            </>
          ) : (
            'Run All Tests'
          )}
        </Button>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              {getStatusIcon(checks.health.status)}
              <span className="ml-2">Health Check</span>
            </CardTitle>
            <CardDescription>
              Tests connection to the FastAPI health endpoint
            </CardDescription>
          </CardHeader>
          <CardContent>
            {renderResult(checks.health)}
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              {getStatusIcon(checks.root.status)}
              <span className="ml-2">Root Endpoint</span>
            </CardTitle>
            <CardDescription>
              Tests the root endpoint that lists all routes
            </CardDescription>
          </CardHeader>
          <CardContent>
            {renderResult(checks.root)}
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              {getStatusIcon(checks.recommendations.status)}
              <span className="ml-2">Portfolio Recommendations</span>
            </CardTitle>
            <CardDescription>
              Tests the FastAPI recommendations endpoint
            </CardDescription>
          </CardHeader>
          <CardContent>
            {renderResult(checks.recommendations)}
          </CardContent>
        </Card>
      </div>
      
      <Tabs defaultValue="inspector">
        <TabsList className="w-full justify-start">
          <TabsTrigger value="inspector">Service Inspector</TabsTrigger>
          <TabsTrigger value="direct">Direct API Test</TabsTrigger>
        </TabsList>
        
        <TabsContent value="inspector" className="mt-4">
          <ServiceInspector />
        </TabsContent>
        
        <TabsContent value="direct" className="mt-4">
          <DirectApiTest />
        </TabsContent>
      </Tabs>
      
      {/* <div className="mt-6 p-4 bg-muted rounded border">
        <h2 className="text-xl font-semibold mb-2">Troubleshooting Guide</h2>
        <ul className="space-y-2 list-disc list-inside">
          <li><strong>FastAPI service not running:</strong> If health check fails, make sure your FastAPI server is running with <code className="bg-muted-foreground/20 p-1 rounded">python -m uvicorn portfolio_service:app --host 0.0.0.0 --port 8001 --reload</code></li>
          <li><strong>Wrong routes:</strong> If health check works but other endpoints fail, check the route paths in your FastAPI app</li>
          <li><strong>CORS issues:</strong> If direct API calls fail from browser, check CORS settings in FastAPI</li>
          <li><strong>Data processing errors:</strong> If endpoints return 500 status, check your backend logs for errors</li>
        </ul>
      </div> */}
    </div>
  );
} 