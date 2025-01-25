// components/share/Navbar.tsx

"use client";

import * as React from "react";
import Link from "next/link";
import { Settings } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import ThemeChanger from "./ThemeChanger";
import { useRouter } from "next/navigation";
import { useWatchlists } from "@/hooks/alpaca/useWatchlists";
import { Watchlist } from "@alpacahq/typescript-sdk";

export default function Navbar() {
  const router = useRouter();

  const handleLogout = async () => {
    await fetch("/api/alpaca/auth/logout", { method: "POST" });
    router.push("/");
  };

  const { watchlists, isLoading, isError, error } = useWatchlists();

  // Render cached data immediately if available
  const cachedWatchlists = typeof window !== "undefined" ? localStorage.getItem("watchlists") : null;
  const initialWatchlists = cachedWatchlists ? JSON.parse(cachedWatchlists) : null;

  return (
    <nav className="bg-background border-b">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex">
            <div className="flex-shrink-0 flex items-center">
              <Link
                href="/"
                className="inline-flex items-center px-1 pt-1 text-sm font-medium text-primary"
              >
                <span className="text-2xl font-bold text-primary">
                  <b>DTF</b>
                </span>
              </Link>
            </div>
            <div className="hidden sm:ml-6 sm:flex sm:space-x-8">
              <Link
                href="/"
                className="inline-flex items-center px-1 pt-1 text-sm font-medium text-primary"
              >
                Dasboard
              </Link>
              <Link
                href="/positions"
                className="inline-flex items-center px-1 pt-1 text-sm font-medium text-muted-foreground hover:text-primary"
              >
                Positions
              </Link>
              <div className="inline-flex items-center px-1 pt-1 text-sm font-medium text-muted-foreground hover:text-primary">
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="outline">
                    Watchlists
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  {isLoading && <div>Loading watchlists...</div>}
                  {isError && <div>Error: {error.message}</div>}
                  {(initialWatchlists || watchlists)?.map((w: Watchlist) =>
                    <Link key={w.id} href={`/assets?watchlist=${w.id}`}>
                    <DropdownMenuItem>
                      {w.name}
                    </DropdownMenuItem>
                    </Link>
                  )}
                  
                </DropdownMenuContent>
              </DropdownMenu>
              </div>
              <Link
                href="#"
                className="inline-flex items-center px-1 pt-1 text-sm font-medium text-muted-foreground hover:text-primary"
              >
                Contact
              </Link>
            </div>
          </div>
          <div className="flex items-center">
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="icon">
                  <Settings className="h-5 w-5" />
                  <span className="sr-only">Open settings menu</span>
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuLabel>Settings</DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuItem>
                  <ThemeChanger />
                </DropdownMenuItem>
                <DropdownMenuItem>Profile</DropdownMenuItem>
                <DropdownMenuItem>Preferences</DropdownMenuItem>
                <DropdownMenuItem>Notifications</DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem onSelect={handleLogout}>
                  Log out
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>
      </div>
    </nav>
  );
}
