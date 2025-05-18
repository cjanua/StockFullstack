/* eslint-disable @typescript-eslint/no-explicit-any */
import { useState, useMemo, useCallback } from 'react';

export type SortDirection = 'asc' | 'desc';

export interface SortConfig {
  key: string;
  direction: SortDirection;
}

/**
 * Custom hook for sorting data in tables or lists
 */
export function useSortableData<T>(
  items: T[],
  initialConfig: SortConfig,
  customCompare?: (a: T, b: T, config: SortConfig) => number
) {
  // Single useState call with an object for stable hook order
  const [sortConfig, setSortConfig] = useState<SortConfig>(initialConfig);

  // Memoize the request sort function to prevent unnecessary re-renders
  const requestSort = useCallback((key: string) => {
    setSortConfig(currentConfig => {
      if (currentConfig.key === key) {
        // Toggle direction
        return {
          key,
          direction: currentConfig.direction === 'asc' ? 'desc' : 'asc',
        };
      }
      // New key, default to ascending
      return { key, direction: 'asc' };
    });
  }, []);

  // Sort the items with useMemo for performance
  const sortedItems = useMemo(() => {
    if (!items || items.length === 0 || !sortConfig.key) {
      return [...(items || [])];
    }

    return [...items].sort((a, b) => {
      // Use custom comparison if provided
      if (customCompare) {
        return customCompare(a, b, sortConfig);
      }
      
      // Default comparison logic
      const valueA = a && typeof a === 'object' && sortConfig.key in a ? (a as any)[sortConfig.key] : null;
      const valueB = b && typeof b === 'object' && sortConfig.key in b ? (b as any)[sortConfig.key] : null;
      
      if (valueA == null && valueB == null) return 0;
      if (valueA == null) return sortConfig.direction === 'asc' ? -1 : 1;
      if (valueB == null) return sortConfig.direction === 'asc' ? 1 : -1;
      
      if (valueA > valueB) return sortConfig.direction === 'asc' ? 1 : -1;
      if (valueA < valueB) return sortConfig.direction === 'asc' ? -1 : 1;
      return 0;
    });
  }, [items, sortConfig, customCompare]);
  
  return { items: sortedItems, requestSort, sortConfig };
}
