// components/Providers.tsx
"use client";

import { ThemeProvider as NextThemesProvider } from "next-themes";
import * as React from "react";
import Navbar from "../components/Navbar";

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  return (
    <NextThemesProvider attribute="class" defaultTheme="system" enableSystem>
      {children}
    </NextThemesProvider>
  );
}

export function NavbarProvider({ children }: { children: React.ReactNode }) {
  return (
    <>
      <Navbar />
      {children}
    </>
  );
}
