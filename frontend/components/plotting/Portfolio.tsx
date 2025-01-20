"use client";

import { TrendingUp } from "lucide-react";
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts";

import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import { usePortfolioHistory } from "@/hooks/alpaca/usePortfolioHistory";
import { fmtCurrency } from "@/lib/utils";


export function AccountGraph() {
  const { portfolioHistory, isLoading, isError, error } = usePortfolioHistory(30);
  if (isLoading) return <div>Loading portfolio history data...</div>;
  if (isError) return error.fallback;
  if (!portfolioHistory) return <div>No portfolio history data available</div>;

  const chartData = portfolioHistory.timestamp.map((t, i) => ({
    date: parseInt(t)*1000,
    value: parseFloat(portfolioHistory.equity[i]),
  }));

  const chartConfig = {
    value: {
      label: "Value",
      color: "hsl(var(--chart-1))",
    },
    date: {
      label: "Date",
      color: "hsl(var(--chart-1))",
    },
  } satisfies ChartConfig;

  const low = Math.min(...chartData.map((d) => Math.floor(d.value)));
  const high = Math.max(...chartData.map((d) => Math.ceil(d.value)));
  const minY = low - (high-low)*.1;
  const maxY = high + (high-low)*.1;

  let prev = "";
  
  return (
    <Card>
      <CardHeader>
        <CardTitle>Account Performance</CardTitle>
        <CardDescription>January - June 2024</CardDescription>
      </CardHeader>
      <CardContent>
        <ChartContainer config={chartConfig}>
          <LineChart
            accessibilityLayer
            data={chartData}
            margin={{
              left: 12,
              right: 12,
            }}
          >
            <CartesianGrid />
            <XAxis
              dataKey="date"
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              tickFormatter={(v) => {
                const value = new Date(v).toLocaleDateString("en-US")
                if (typeof value === "string") {
                  const parts = value.split("/");
                  let build = parts[0] + "/" + parts[1]
                  if (prev !== parts[2]) {
                    build += "/" + parts[2].slice(2, 4)
                    prev = parts[2]
                  }
                  prev = parts[2]
                  return build
                }
                return value;
              }}
              // tickFormatter={(value) => value.slice(0, 3)}
            />
            <YAxis
              domain={[minY, maxY]}
              tickFormatter={(value: number | string) => {
                if (typeof value === "string") {
                  return fmtCurrency(parseFloat(value))
                }
                if (typeof value === "number") {
                  return fmtCurrency(value)
                }
                return value;
              }}
            />
            <ChartTooltip
              cursor={false}
              content={<ChartTooltipContent />}
              formatter={(value: number) => [`${fmtCurrency(value)} USD`, ""]}
            />
            <Line
              dataKey="value"
              type="stepAfter"
              stroke="var(--color-value)"
              strokeWidth={2}
              dot={false}
            />
          </LineChart>
        </ChartContainer>
      </CardContent>
      <CardFooter className="flex-col items-start gap-2 text-sm">
        <div className="flex gap-2 font-medium leading-none">
          Trending up by 5.2% this month <TrendingUp className="h-4 w-4" />
        </div>
        <div className="leading-none text-muted-foreground">
          Showing total visitors for the last 6 months
        </div>
      </CardFooter>
    </Card>
  );
}
