import { Loader2 } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { fmtCurrency, fmtShares } from '@/lib/utils';
import { ActionButtonProps } from '@/types/order';

export function ActionButton({
  action,
  quantity,
  price,
  // symbol,
  onClick,
  isExecuting = false
}: ActionButtonProps) {
  const estimatedValue = quantity * price;
  const colorClass = action === 'Buy' ? 'border-green-500 text-green-500' : 'border-red-500 text-red-500';
  
  return (
    <Badge 
      variant={action === 'Buy' ? "outline" : "secondary"} 
      className={`
        ${colorClass}
        ml-auto inline-flex w-24 justify-end cursor-pointer hover:opacity-80
      `}
      onClick={onClick}
    >
      {isExecuting ? (
        <Loader2 className="h-4 w-4 animate-spin" />
      ) : (
        <div className="flex flex-col items-end group">
          <span className="group-hover:hidden">{fmtCurrency(estimatedValue)}</span>
          <span className="hidden group-hover:block">{action} {fmtShares(quantity)}</span>
        </div>
      )}
    </Badge>
  );
}
