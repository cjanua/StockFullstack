// components/alpaca/PortfolioRecommendations.tsx
'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Loader2, TrendingUp, RefreshCw, AlertTriangle } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';

interface PortfolioRecommendation {
  symbol: string;
  current_shares: number;
  target_shares: number;
  difference: number;
  action: 'Buy' | 'Sell';
  quantity: number;
}

interface RecommendationResponse {
  portfolio_value: number;
  cash: number;
  target_cash: number;
  recommendations: PortfolioRecommendation[];
}

export function PortfolioRecommendations() {
  const [lookbackDays, setLookbackDays] = useState(365);
  
  const { data, isLoading, isError, error, refetch } = useQuery<RecommendationResponse>({
    queryKey: ['portfolioRecommendations', lookbackDays],
    queryFn: async () => {
      const response = await axios.get('/api/alpaca/portfolio/recommendations', {
        params: {
          lookback_days: lookbackDays,
          min_change_percent: 0.01,
          cash_reserve_percent: 0.05
        }
      });
      return response.data;
    },
    enabled: true,
    refetchOnWindowFocus: false,
  });
  
  const buyRecommendations = data?.recommendations.filter(r => r.action === 'Buy') || [];
  const sellRecommendations = data?.recommendations.filter(r => r.action === 'Sell') || [];
  
  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex justify-between items-center">
          <div>
            <CardTitle className="flex items-center">
              <TrendingUp className="mr-2 h-5 w-5" />
              Portfolio Optimization
            </CardTitle>
            <CardDescription>
              Recommended trades to optimize your portfolio
            </CardDescription>
          </div>
          <Button 
            variant="outline" 
            size="sm" 
            onClick={() => refetch()} 
            disabled={isLoading}
          >
            {isLoading ? (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" /> 
            ) : (
              <RefreshCw className="mr-2 h-4 w-4" />
            )}
            Refresh
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="flex justify-center items-center p-8">
            <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
          </div>
        ) : isError ? (
          <Alert variant="destructive">
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>
              {error instanceof Error 
                ? error.message 
                : "Unable to get portfolio recommendations. Please try again later or check if the portfolio service is running."}
            </AlertDescription>
            <div className="mt-2">
              <Button variant="outline" size="sm" onClick={() => refetch()}>
                Try Again
              </Button>
            </div>
          </Alert>
        ) : (
          <>
            <div className="grid grid-cols-3 gap-4 mb-6">
              <div className="bg-muted/50 p-3 rounded-lg">
                <div className="text-sm text-muted-foreground">Portfolio Value</div>
                <div className="text-xl font-bold mt-1">${data?.portfolio_value.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</div>
              </div>
              <div className="bg-muted/50 p-3 rounded-lg">
                <div className="text-sm text-muted-foreground">Current Cash</div>
                <div className="text-xl font-bold mt-1">${data?.cash.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</div>
              </div>
              <div className="bg-muted/50 p-3 rounded-lg">
                <div className="text-sm text-muted-foreground">Target Cash</div>
                <div className="text-xl font-bold mt-1">${data?.target_cash.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</div>
              </div>
            </div>
            
            <Tabs defaultValue="buy">
              <TabsList className="mb-4">
                <TabsTrigger value="buy">
                  Buy Recommendations
                  {buyRecommendations.length > 0 && (
                    <Badge variant="secondary" className="ml-2">
                      {buyRecommendations.length}
                    </Badge>
                  )}
                </TabsTrigger>
                <TabsTrigger value="sell">
                  Sell Recommendations
                  {sellRecommendations.length > 0 && (
                    <Badge variant="secondary" className="ml-2">
                      {sellRecommendations.length}
                    </Badge>
                  )}
                </TabsTrigger>
              </TabsList>
              
              <TabsContent value="buy">
                {buyRecommendations.length > 0 ? (
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Symbol</TableHead>
                        <TableHead>Current Shares</TableHead>
                        <TableHead>Target Shares</TableHead>
                        <TableHead className="text-right">Buy Quantity</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {buyRecommendations.map((rec) => (
                        <TableRow key={rec.symbol}>
                          <TableCell className="font-medium">{rec.symbol}</TableCell>
                          <TableCell>{rec.current_shares.toFixed(0)}</TableCell>
                          <TableCell>{rec.target_shares.toFixed(0)}</TableCell>
                          <TableCell className="text-right font-bold text-green-600">
                            {rec.quantity.toFixed(0)}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                ) : (
                  <div className="text-center py-8 text-muted-foreground">
                    No buy recommendations at this time
                  </div>
                )}
              </TabsContent>
              
              <TabsContent value="sell">
                {sellRecommendations.length > 0 ? (
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Symbol</TableHead>
                        <TableHead>Current Shares</TableHead>
                        <TableHead>Target Shares</TableHead>
                        <TableHead className="text-right">Sell Quantity</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {sellRecommendations.map((rec) => (
                        <TableRow key={rec.symbol}>
                          <TableCell className="font-medium">{rec.symbol}</TableCell>
                          <TableCell>{rec.current_shares.toFixed(0)}</TableCell>
                          <TableCell>{rec.target_shares.toFixed(0)}</TableCell>
                          <TableCell className="text-right font-bold text-red-600">
                            {rec.quantity.toFixed(0)}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                ) : (
                  <div className="text-center py-8 text-muted-foreground">
                    No sell recommendations at this time
                  </div>
                )}
              </TabsContent>
            </Tabs>
          </>
        )}
      </CardContent>
    </Card>
  );
}