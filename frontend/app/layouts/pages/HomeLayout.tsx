// components/pages/home/AccountOverview.tsx

"use client";
import { useAccount } from "@/hooks/alpaca/useAccount";
import { formatCurrency, formatDate, isValidDate } from "@/lib/utils";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";

export function AccountOverview() {
  const { account, isLoading, error } = useAccount();

  if (isLoading) return <div>Loading account data...</div>;
  if (error) return <div>Error: {error}</div>;
  if (!account) return <div>No account data available</div>;

  const equityPercentage =
    (parseFloat(account.equity) / parseFloat(account.portfolio_value)) * 100;

  return (
    <div className="container mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6">Financial Dashboard</h1>

      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              Account Balance
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {formatCurrency(parseFloat(account.portfolio_value))}
            </div>
            <p className="text-xs text-muted-foreground">
              Account #{account.account_number}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Buying Power</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {formatCurrency(parseFloat(account.buying_power))}
            </div>
            <p className="text-xs text-muted-foreground">
              Available for trading
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Equity</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {formatCurrency(parseFloat(account.equity))}
            </div>
            <Progress value={equityPercentage} className="mt-2" />
            <p className="text-xs text-muted-foreground mt-1">
              {equityPercentage.toFixed(2)}% of portfolio value
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Cash Balance</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {formatCurrency(parseFloat(account.cash))}
            </div>
            <p className="text-xs text-muted-foreground">
              Available for withdrawal
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              Long Market Value
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {formatCurrency(parseFloat(account.long_market_value))}
            </div>
            <p className="text-xs text-muted-foreground">
              Value of long positions
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              Account Status
            </CardTitle>
          </CardHeader>
          <CardContent>
            <Badge>{account.status.split(".")[1]}</Badge>
            <p className="text-xs text-muted-foreground mt-2">
              Day trades: {account.daytrade_count}
            </p>
            <p className="text-xs text-muted-foreground">
              Created:{" "}
              {isValidDate(account.created_at)
                ? formatDate(account.created_at)
                : "N/A"}
            </p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
