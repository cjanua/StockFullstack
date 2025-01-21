import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts";

import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import { JSX } from "react";
import { NameType, Payload, ValueType } from "recharts/types/component/DefaultTooltipContent";

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

type Point = {
  date: number;
  value: number;
}


export default function PortfolioHistoryGraph(
  {chartData, fmtXAxis, fmtYAxis, fmtTooltip} : 
  {
    chartData: Point[],
    fmtXAxis?: ({ x, y, payload }: { x: number, y: number, payload: Point }) => JSX.Element,
    fmtYAxis?: (value: number, index: number ) => string,
    fmtTooltip?: (_label: unknown, payload: Payload<ValueType, NameType>[]) => JSX.Element
  }
) {
    const low = Math.min(...chartData.map((d: Point) => Math.floor(d.value)));
    const high = Math.max(...chartData.map((d: Point) => Math.ceil(d.value)));
    const minY = low - (high-low)*.1;
    const maxY = high + (high-low)*.1;

    return (
    <ChartContainer config={chartConfig}>
      <LineChart
        accessibilityLayer
        data={chartData}
        margin={{
          left: 12,
          right: 12,
          bottom: 2
        }}
      >
        <CartesianGrid />
        <XAxis
          dataKey="date"
          tickLine={false}
          axisLine={false}
          tickMargin={8}
          tick={fmtXAxis}
        />
        <YAxis
          domain={[minY, maxY]}
          tickFormatter={fmtYAxis}
        />
        <ChartTooltip
          cursor={false}
          content={<ChartTooltipContent />}
          labelFormatter={fmtTooltip}
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
    )
}