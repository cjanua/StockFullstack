"use client";

import {useState, useRef, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  TrendingUp,
  ShoppingCart,
  List,
  Settings,
  Search,
  Activity
} from "lucide-react";

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
import { useWatchlists } from "@/hooks/queries/useAlpacaQueries";
import { Watchlist } from "@/types/alpaca";
import { cn } from "@/lib/utils";
import { useOnClickOutside } from "@/hooks/useOnClickOutside";

export default function Sidebar() {
  const [isExpanded, setIsExpanded] = useState(false);
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const sidebarRef = useRef<HTMLDivElement>(null!);
  const pathname = usePathname();
  const { data: watchlists, isLoading, isError, error } = useWatchlists();

  // Render cached data immediately if available
  const cachedWatchlists = typeof window !== "undefined" ? localStorage.getItem("watchlists") : null;
  const initialWatchlists = cachedWatchlists ? JSON.parse(cachedWatchlists) : null;

  const navItems = [
    { name: "Dashboard", href: "/", icon: <LayoutDashboard className="h-5 w-5" /> },
    { name: "Positions", href: "/positions", icon: <TrendingUp className="h-5 w-5" /> },
    { name: "Orders", href: "/orders", icon: <ShoppingCart className="h-5 w-5" /> },
    { name: "Health", href: "/api-diagnostics", icon: <Activity className="h-5 w-5" /> },
    // { name: "Contact", href: "#", icon: <Contact className="h-5 w-5" /> },
  ];

  // Handle clicking outside of sidebar
  useOnClickOutside(sidebarRef, () => {
    if (isExpanded && !dropdownOpen) {
      setIsExpanded(false);
    }
  });

  // Handle mouse leave only when no dropdown is open
  const handleMouseLeave = () => {
    if (!dropdownOpen) {
      setIsExpanded(false);
    }
  };

  useEffect(() => {
    setIsExpanded(false);
  }, [pathname]);

  return (
    <div
      ref={sidebarRef}
      className={cn(
        "fixed left-0 top-0 h-full bg-background border-r transition-all duration-300 ease-in-out z-40",
        isExpanded ? "w-56" : "w-16",
        isExpanded ? "sidebar-expanded" : "sidebar-collapsed"
      )}
      onMouseEnter={() => setIsExpanded(true)}
      onMouseLeave={handleMouseLeave}
    >
      <div className="flex flex-col h-full">
        <div className="p-4">
          <Link href="/" className="flex items-center justify-center">
            <span className={cn("text-2xl font-bold text-primary transition-opacity",
              isExpanded ? "opacity-100" : "opacity-0 hidden"
            )}>
              <b>DTF</b>
            </span>
            <span className={cn("text-2xl font-bold text-primary",
              !isExpanded ? "opacity-100" : "opacity-0 hidden"
            )}>
              <b>D</b>
            </span>
          </Link>
        </div>

        <nav className="flex-1 pt-5">
          <ul className="space-y-2 px-2">
            {navItems.map((item) => (
              <li key={item.name}>
                <Link
                  href={item.href}
                  className={cn(
                    "flex items-center rounded-md p-2 transition-all",
                    pathname === item.href ? "bg-secondary text-primary" : "hover:bg-secondary/50 text-muted-foreground hover:text-primary"
                  )}
                >
                  <div className="flex items-center">
                    {item.icon}
                    <span className={cn("ml-3 transition-opacity",
                      isExpanded ? "opacity-100" : "opacity-0 hidden"
                    )}>
                      {item.name}
                    </span>
                  </div>
                </Link>
              </li>
            ))}

            <li>
              <DropdownMenu onOpenChange={(open) => setDropdownOpen(open)}>
                <DropdownMenuTrigger asChild>
                  <Button variant="ghost" className={cn(
                    "flex w-full justify-start rounded-md p-2",
                    "hover:bg-secondary/50 text-muted-foreground hover:text-primary"
                  )}>
                    <List className="h-5 w-5" />
                    <span className={cn("ml-3 transition-opacity",
                      isExpanded ? "opacity-100" : "opacity-0 hidden"
                    )}>
                      Watchlists
                    </span>
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent 
                  align="end" 
                  side="right" 
                  className="z-50"
                >
                  {/* {isLoading && <div>Loading watchlists...</div>}
                  {isError && <div>Error: {error.message}</div>} */}
                  {((Array.isArray(initialWatchlists) ? initialWatchlists : []) || (Array.isArray(watchlists) ? watchlists : []))?.map((w: Watchlist) => (
                    <Link key={w.id} href={`/assets?watchlist=${w.id}`}>
                      <DropdownMenuItem>{w.name}</DropdownMenuItem>
                    </Link>
                  ))}
                </DropdownMenuContent>
              </DropdownMenu>
            </li>
          </ul>
        </nav>

        <div className="mt-auto p-4 space-y-2">
          <Button variant="ghost" className="w-full justify-start rounded-md p-2">
            <Search className="h-5 w-5" />
            <span className={cn("ml-3 transition-opacity",
              isExpanded ? "opacity-100" : "opacity-0 hidden"
            )}>
              Search
            </span>
          </Button>

          <DropdownMenu onOpenChange={(open) => setDropdownOpen(open)}>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" className="w-full justify-start rounded-md p-2">
                <Settings className="h-5 w-5" />
                <span className={cn("ml-3 transition-opacity",
                  isExpanded ? "opacity-100" : "opacity-0 hidden"
                )}>
                  Settings
                </span>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent 
              align="end" 
              side="right"
              className="z-50"
            >
              <DropdownMenuLabel>Settings</DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem>
                <ThemeChanger />
              </DropdownMenuItem>
              <Link href="/account">
                <DropdownMenuItem>Profile</DropdownMenuItem>
              </Link>
              <DropdownMenuItem>Preferences</DropdownMenuItem>
              <DropdownMenuItem>Notifications</DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>
    </div>
  );
}
