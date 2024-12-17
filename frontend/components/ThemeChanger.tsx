// components/share/ThemeChanger.tsx

"use client";
import { useTheme } from "next-themes";
import { useEffect, useState } from "react";

import { Moon, Sun } from "lucide-react";
import { Switch } from "./ui/switch";

// Need to handle hydration mismatch
const ThemeChanger = () => {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const handleToggle = () => {
    const newState = theme == "light" ? "dark" : "light";
    setTheme(newState);
  };

  if (!mounted) return null;

  return (
    <div className="flex items-center space-x-2">
      <Sun className="h-4 w-4" />
      <Switch
        checked={theme == "dark"}
        onCheckedChange={handleToggle}
        aria-label="Toggle theme"
      />
      <Moon className="h-4 w-4" />
    </div>
  );
};

export default ThemeChanger;
