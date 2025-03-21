import { Position } from './alpaca';

export interface OrderAction {
  symbol: string;
  action: 'Buy' | 'Sell';
  quantity: number;
  price?: number;
}

export interface OrderDialogProps {
  symbol: string;
  action: 'Buy' | 'Sell';
  quantity: number;
  price: number;
  // These props are now handled by the parent AlertDialog
  isOpen?: boolean;
  onOpenChange?: (open: boolean) => void;
  onExecute: (symbol: string, action: 'Buy' | 'Sell', quantity: number) => void;
  isExecuting: boolean;
  availableCash?: number;
}

// For badge/button rendering in tables
export interface ActionButtonProps {
  action: 'Buy' | 'Sell';
  quantity: number;
  price: number;
  symbol: string;
  onClick?: () => void;
  isExecuting?: boolean;
}
