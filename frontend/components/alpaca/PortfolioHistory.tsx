"use client";

import { TrendingDown, TrendingUp } from "lucide-react";

import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

import { toast } from "@/hooks/use-toast";
import { Button } from "@/components/ui/button"
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";

import { usePortfolioHistory } from "@/hooks/alpaca/usePortfolioHistory";
import { fmtCurrency, fmtPercent } from "@/lib/utils";
import { Select, SelectContent, SelectGroup, SelectItem, SelectLabel, SelectTrigger, SelectValue } from "../ui/select";
import { z } from "zod"
import { zodResolver } from "@hookform/resolvers/zod"

import { useForm } from "react-hook-form"
import { useState, useEffect } from "react";
import PortfolioHistoryGraph from "../plotting/PortfolioHistoryGraph";

import { NameType, Payload, ValueType } from "recharts/types/component/DefaultTooltipContent";
import { Skeleton } from "../ui/skeleton";


const daysOptions = [
  { value: "2D", display: "1D"},
  { value: "7D", display: "1W"},
  { value: "30D", display: "1M"},
  { value: "180D", display: "6M"},
  { value: "365D", display: "1Y"},
  { value: "-1D", display: "All Time"}
]

const timeframeOptionsBase = [
  { value: "5Min", display: "5m"},
  { value: "15Min", display: "15m"},
  { value: "1H", display: "1h"},
  { value: "1D", display: "1D"},
]

const FormSchema = z.object({
  days: z.string({ required_error: "Days is required" }),
  timeframe: z.string({ required_error: "Timeframe is required" }),
})

export function AccountGraph() {
  const [timeframeOptions, setTimeframeOptions] = useState(timeframeOptionsBase);
  const [tmpDays, setTmpDays] = useState("180D");

  useEffect(() => {
    const numDays = parseInt(tmpDays.slice(0, -1))
    if (numDays >= 30)
      setTimeframeOptions(timeframeOptionsBase.filter((option) => (!option.value.endsWith("Min")) && !option.value.endsWith("H")));
    else if (numDays >= 7)
      setTimeframeOptions(timeframeOptionsBase.filter((option) => option.value !== "1Min"));
    else
      setTimeframeOptions(timeframeOptionsBase);
  }, [tmpDays]);

  const [days, setDays] = useState(180);
  const [timeframe, setTimeframe] = useState("1D");

  useEffect(() => {
    // This will trigger the usePortfolioHistory hook to refetch data when days changes
  }, [days, timeframe]);

  const form = useForm<z.infer<typeof FormSchema>>({
    resolver: zodResolver(FormSchema),
    defaultValues: {
      days: `${days}D`,
      timeframe: timeframe,
    }
  });

  const { portfolioHistory, isLoading, isError, error } = usePortfolioHistory(days, timeframe);

  if (isLoading) return <>
    <div className="flex flex-col space-y-3 pb-8">
      <div className="space-y-2">
        <Skeleton className="h-4 w-[250px]" />
        <Skeleton className="h-4 w-[200px]" />
      </div>
    </div>
  </>;
  if (isError) return error.fallback;
  if (!portfolioHistory) return <div>No portfolio history data available</div>;

  let chartData = portfolioHistory.timestamp.map((t, i) => ({
    date: parseInt(t)*1000,
    value: parseFloat(portfolioHistory.equity[i]),
  }));
  chartData = chartData.filter((d) => d.value > 0);
  chartData = chartData.sort((a, b) => a.date - b.date);

  function onSubmit(data: z.infer<typeof FormSchema>) {
    setDays(parseInt(data.days.slice(0, -1)) ?? days);
    setTimeframe(data.timeframe ?? timeframe);
    toast({
      title: "You submitted the following values:",
      description: (
        <pre className="mt-2 w-[340px] rounded-md bg-slate-950 p-4">
          <code className="text-white">{JSON.stringify(data, null, 2)}</code>
        </pre>
      ),
    })
  }

  let prev = "";
  const fmtXAxis = ({ x, y, payload }: { x: number, y: number, payload: {value: number, date: number}}) => {
    let mainValue;
    let subValue;
    let render = false;

    const date = new Date(payload.value)

    if (timeframe.endsWith("D")) {
      mainValue = date.toLocaleString("en-US", {
        month: "2-digit",
        day: "2-digit",
      });
      subValue =  date.toLocaleString("en-US", {
        year: "2-digit",
      });
  
    } else {
      subValue = date.toLocaleDateString("en-US", {
        month: "2-digit",
        day: "2-digit"
      });
      mainValue = date.toLocaleTimeString("en-US", {
        hour: "2-digit",
        minute: "2-digit"
      });
    }
    render = prev !== subValue;
    prev = subValue;
    return (
      <svg>
      <foreignObject x={x - 40} y={y - 10} width={80} height={40}>
        <div className="flex flex-col items-center">
          <span>{mainValue}</span>
          {render && <span>{subValue}</span>}
        </div>
      </foreignObject>
    </svg>
    );
  };

  const fmtYAxis = (value: string | number) => {
    if (typeof value === "string") {
      return fmtCurrency(parseFloat(value))
    }
    if (typeof value === "number") {
      return fmtCurrency(value)
    }
    return value;
  }

  const fmtTooltip = (_label: unknown, payload: Payload<ValueType, NameType>[]) => {
    const p = payload[0].payload;

    const date = new Date(p.date).toLocaleDateString();
    if (timeframe.endsWith("D"))
      return <>{date}</>;
    const time = new Date(p.date).toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" });
    return <>{date} {time}</>;
  }

  const ratioDiff = ((chartData.at(-1)?.value ?? 0) - (chartData.at(0)?.value ?? 0)) / (chartData.at(0)?.value ?? 1);
  
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-4">
        <div>
          <CardTitle>Account Performance</CardTitle>
          <CardDescription>January - June 2024</CardDescription>
        </div>
        <div className="flex flex-row items-center justify-between gap-1">
          <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="w-2/3 space-y-6 flex flex-row items-end justify-between gap-1">
            <FormField
              control={form.control}
              name="days"
              render={({ field }) => (
              <FormItem>
                <FormLabel>Days</FormLabel>
                  <Select onValueChange={(v: string) => {field.onChange(v); setTmpDays(v)}} defaultValue={tmpDays}>
                    <FormControl>
                      <SelectTrigger className="w-[180px]">
                        <SelectValue placeholder="Select Days" />
                      </SelectTrigger>
                    </FormControl>
                    <SelectContent>
                      <SelectGroup>
                        <SelectLabel>Days</SelectLabel>
                        {daysOptions.map((option) => 
                          <SelectItem
                            key={option.value}
                            value={option.value}
                          >
                            {option.display}
                          </SelectItem>
                        )}
                      </SelectGroup>
                    </SelectContent>
                  </Select>
                <FormMessage />
              </FormItem>
            )} />
            <FormField
              control={form.control}
              name="timeframe"
              render={({ field }) => (
              <FormItem>
                <FormLabel>Timeframe</FormLabel>
                  <Select onValueChange={field.onChange} defaultValue={timeframe} value={timeframe}>
                    <FormControl>
                      <SelectTrigger className="w-[180px]">
                        <SelectValue placeholder="Select timeframe" />
                      </SelectTrigger>
                    </FormControl>            
                    <SelectContent>
                      <SelectGroup>
                        <SelectLabel>Timeframe</SelectLabel>
                        {timeframeOptions.map((option) => 
                          <SelectItem
                            key={option.value}
                            value={option.value}
                          >
                            {option.display}
                          </SelectItem>
                        )}
                      </SelectGroup>
                    </SelectContent>
                  </Select>
                <FormMessage />
              </FormItem>
            )} />
            <Button type="submit">Submit</Button>
          </form>
          </Form>
        </div>
      </CardHeader>
      <CardContent>
        <PortfolioHistoryGraph chartData={chartData} fmtXAxis={fmtXAxis} fmtYAxis={fmtYAxis} fmtTooltip={fmtTooltip}/>
      </CardContent>
      <CardFooter className="flex-col items-start gap-2 text-sm">
        <div className="flex gap-2 font-medium leading-none">
          Trending up by {fmtPercent(ratioDiff)} in {days != -1 ? days : "All"} Days
          {ratioDiff > 0 ? <TrendingUp className="h-4 w-4" /> : <TrendingDown className="h-4 w-4" />}
        </div>
        <div className="leading-none text-muted-foreground">
          Showing total visitors for the last 6 months
        </div>
      </CardFooter>
    </Card>
  );
}
