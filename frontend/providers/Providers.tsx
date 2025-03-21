// components/Providers.tsx
"use client";

import { ThemeProvider as NextThemesProvider } from "next-themes";
import Sidebar from "../components/Sidebar";
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactNode, useState } from 'react';
import { usePathname } from "next/navigation";
import { publicPaths } from "@/middleware";


export function ThemeProvider({ children }: { children: ReactNode }) {
  return (
    <NextThemesProvider attribute="class" defaultTheme="system" enableSystem>
      {children}
    </NextThemesProvider>
  );
}

export function NavbarProvider({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const isAuthPage = publicPaths.some(path => pathname.startsWith(path));

  if (isAuthPage) {
    return <div className="auth-layout">{children}</div>;
  }

  return (
    <div className="flex">
      <Sidebar />
      <div className="flex-1 ml-16 transition-all duration-300 ease-in-out">
        {children}
      </div>
    </div>
  );
}

export function QueryProvider({ children }: { children: ReactNode }) {
  const [queryClient] = useState(() => new QueryClient());

  return (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  );
} 