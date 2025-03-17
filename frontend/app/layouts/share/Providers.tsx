// components/Providers.tsx
"use client";

import { ThemeProvider as NextThemesProvider } from "next-themes";
import * as React from "react";
import Sidebar from "../../../components/Sidebar";

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  return (
    <NextThemesProvider attribute="class" defaultTheme="system" enableSystem>
      {children}
    </NextThemesProvider>
  );
}

export function NavbarProvider({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex">
      <Sidebar />
      <div className="ml-16 w-full">
        {children}
      </div>
    </div>
  );
}
