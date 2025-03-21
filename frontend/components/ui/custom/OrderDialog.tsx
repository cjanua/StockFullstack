import { Loader2 } from 'lucide-react';
import { fmtCurrency, fmtCurrencyPrecise, fmtShares } from '@/lib/utils';
import { 
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import { OrderDialogProps } from '@/types/order';

// Change to only render the dialog content, not the full AlertDialog
export function OrderDialog({
  symbol,
  action,
  quantity,
  price,
  isExecuting,
  availableCash,
  onExecute
}: OrderDialogProps) {
  const estimatedValue = quantity * price;
  const actionTextLower = action.toLowerCase();
  const actionText = action === 'Buy' ? 'purchase' : 'sale';
  const colorClass = action === 'Buy' ? 'text-green-500' : 'text-red-500';
  const buttonColorClass = action === 'Buy' ? 'bg-green-500 hover:bg-green-600' : 'bg-red-500 hover:bg-red-600';
  
  // Check if this would exceed available cash (for Buy orders)
  const exceedsCash = action === 'Buy' && availableCash !== undefined && estimatedValue > availableCash;

  const handleExecute = () => {
    onExecute(symbol, action, quantity);
  };

  return (
    <AlertDialogContent>
      <AlertDialogHeader>
        <AlertDialogTitle>{action} {symbol} Shares</AlertDialogTitle>
        <AlertDialogDescription>
          This will create a market order to {actionTextLower} {fmtShares(quantity)} shares of {symbol}.
        </AlertDialogDescription>
        
        <div className="mt-4 p-3 bg-muted rounded-md">
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div>Symbol:</div>
            <div className="font-medium text-right">{symbol}</div>
            
            <div>Current Price:</div>
            <div className="font-medium text-right">{fmtCurrencyPrecise(price)}</div>
            
            <div>Quantity:</div>
            <div className="font-medium text-right">{fmtShares(quantity)} shares</div>
            
            <div className="font-medium">Estimated {actionText}:</div>
            <div className={`font-medium text-right ${colorClass}`}>
              {fmtCurrency(estimatedValue)}
            </div>
            
            {action === 'Buy' && availableCash !== undefined && (
              <>
                <div>Available Cash:</div>
                <div className="font-medium text-right">
                  {fmtCurrency(availableCash)}
                </div>
                {exceedsCash && (
                  <div className="col-span-2 text-red-500 text-xs">
                    Warning: This purchase may exceed your available cash. 
                    The order might be rejected.
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </AlertDialogHeader>
      <AlertDialogFooter>
        <AlertDialogCancel>Cancel</AlertDialogCancel>
        <AlertDialogAction
          onClick={handleExecute}
          disabled={isExecuting}
          className={buttonColorClass}
        >
          {isExecuting ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Submitting...
            </>
          ) : (
            `${action} Shares`
          )}
        </AlertDialogAction>
      </AlertDialogFooter>
    </AlertDialogContent>
  );
}
