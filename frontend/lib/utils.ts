// lib/utils.ts
import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatCurrency(amount: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
  }).format(amount);
}

// Helper function to format dates from Alpaca
export function formatDate(dateString: string | Date): string {
  const date =
    typeof dateString === "string" ? new Date(dateString) : dateString;
  return date.toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

// Helper function to check if a date string is valid
export function isValidDate(dateString: string): boolean {
  const date = new Date(dateString);
  return date instanceof Date && !isNaN(date.getTime());
}

export function fmtCurrency(amt: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
  }).format(amt);
}

/**
 * Format a currency value with specific decimal precision based on amount
 * @param amt The amount to format
 * @param decimals Optional specific number of decimals to use
 * @returns Formatted currency string
 */
export function fmtCurrencyPrecise(amt: number, decimals?: number): string {
  // If explicit decimals are provided, use them
  if (decimals !== undefined) {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    }).format(amt);
  }

  // Otherwise determine appropriate precision based on amount
  if (amt < 0.1) {
    // More precision for penny stocks
    return fmtCurrency(amt);
  } else if (amt < 1) {
    // 3 decimal places for sub-dollar prices
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 3,
      maximumFractionDigits: 3,
    }).format(amt);
  } else if (amt >= 1000) {
    // No decimals for large amounts
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amt);
  } 
  
  // Default currency formatting for normal ranges
  return fmtCurrency(amt);
}

export function fmtPercent(amt: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "percent",
    minimumFractionDigits: 2,
  }).format(amt);
}

/**
 * Format share quantities with appropriate precision
 * @param quantity Share quantity to format
 * @returns Formatted string with appropriate precision
 */
export function fmtShares(quantity: number | string): string {
  const num = typeof quantity === 'string' ? parseFloat(quantity) : quantity;
  
  // For zero value, just return "0.00"
  if (num === 0) {
    return "0.00";
  }
  
  const absNum = Math.abs(num);
  const numStr = num.toString();
  
  // Determine how many significant decimal places to show
  if (absNum < 0.0001) {
    // For extremely small values, find the first non-zero digit
    const decimalStr = numStr.split('.')[1] || '';
    let significantDigits = 2;
    
    // Count leading zeros after decimal point
    let leadingZeros = 0;
    for (let i = 0; i < decimalStr.length; i++) {
      if (decimalStr[i] === '0') {
        leadingZeros++;
      } else {
        break;
      }
    }
    
    // Show at least the first two non-zero digits
    const digitsToShow = Math.max(leadingZeros + 2, 5);
    return num.toFixed(digitsToShow);
  }
  
  if (absNum < 0.01) {
    // Small values, show enough precision to see the actual value
    return num.toFixed(5);
  }
  
  if (absNum < 0.1) {
    // Medium small values
    return num.toFixed(4);
  }
  
  // Standard values (≥0.1)
  return num.toFixed(2);
}
