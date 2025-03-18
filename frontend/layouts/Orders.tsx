"use client";
import { useEffect, useState } from "react";
import { OrderForm } from "@/components/alpaca/OrderForm";
import { 
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle 
} from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
// import { Reload, AlertCircle } from "lucide-react";
import { toast } from "@/hooks/use-toast";

type Order = {
  id: string;
  client_order_id: string;
  created_at: string;
  updated_at: string;
  submitted_at: string;
  filled_at: string | null;
  expired_at: string | null;
  canceled_at: string | null;
  failed_at: string | null;
  asset_id: string;
  symbol: string;
  asset_class: string;
  qty: string;
  filled_qty: string;
  type: string;
  side: string;
  time_in_force: string;
  limit_price: string | null;
  stop_price: string | null;
  filled_avg_price: string | null;
  status: string;
  extended_hours: boolean;
  trail_percent: string | null;
  trail_price: string | null;
  hwm: string | null;
};

export function OrdersLayout() {
  const [orders, setOrders] = useState<Order[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState<'open' | 'closed' | 'all'>('open');

  const fetchOrders = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(`/api/alpaca/orders?status=${statusFilter}`);
      
      if (!response.ok) {
        throw new Error('Failed to fetch orders');
      }
      
      const data = await response.json();
      setOrders(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchOrders();
  }, [statusFilter]);

  const handleCancelOrder = async (orderId: string) => {
    try {
      const response = await fetch(`/api/alpaca/orders?id=${orderId}`, {
        method: 'DELETE',
      });
      
      if (!response.ok) {
        throw new Error('Failed to cancel order');
      }
      
      toast({
        title: "Order Canceled",
        description: `Order ${orderId} has been canceled.`,
      });
      
      // Refresh the orders list
      fetchOrders();
    } catch (err) {
      toast({
        title: "Error",
        description: err instanceof Error ? err.message : 'Failed to cancel order',
        variant: "destructive",
      });
    }
  };

  const handleCancelAllOrders = async () => {
    if (!confirm('Are you sure you want to cancel all open orders?')) {
      return;
    }
    
    try {
      const response = await fetch(`/api/alpaca/orders`, {
        method: 'DELETE',
      });
      
      if (!response.ok) {
        throw new Error('Failed to cancel all orders');
      }
      
      toast({
        title: "Orders Canceled",
        description: "All open orders have been canceled.",
      });
      
      // Refresh the orders list
      fetchOrders();
    } catch (err) {
      toast({
        title: "Error",
        description: err instanceof Error ? err.message : 'Failed to cancel orders',
        variant: "destructive",
      });
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'new':
        return <Badge className="bg-blue-500">New</Badge>;
      case 'filled':
        return <Badge className="bg-green-500">Filled</Badge>;
      case 'partially_filled':
        return <Badge className="bg-amber-500">Partially Filled</Badge>;
      case 'canceled':
        return <Badge variant="outline">Canceled</Badge>;
      case 'expired':
        return <Badge variant="outline" className="bg-slate-700">Expired</Badge>;
      case 'pending_new':
        return <Badge className="bg-blue-400">Pending</Badge>;
      case 'pending_cancel':
        return <Badge className="bg-orange-400">Canceling</Badge>;
      default:
        return <Badge>{status}</Badge>;
    }
  };

  const formatDate = (dateString: string | null) => {
    if (!dateString) return '-';
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  return (
    <div className="container mx-auto p-6">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold">Orders</h1>
        <div className="flex gap-4">
          <OrderForm />
          <Button 
            variant="outline" 
            onClick={fetchOrders}
            disabled={isLoading}
          >
            {/* <Reload className="h-4 w-4 mr-2" /> */}
            Refresh
          </Button>
        </div>
      </div>

      <Card className="mb-6">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <div>
            <CardTitle>Order Management</CardTitle>
            <CardDescription>View and manage your orders</CardDescription>
          </div>
          <div className="flex items-center gap-4">
            <Select 
              value={statusFilter} 
              onValueChange={(value: 'open' | 'closed' | 'all') => setStatusFilter(value)}
            >
              <SelectTrigger className="w-36">
                <SelectValue placeholder="Filter by status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="open">Open Orders</SelectItem>
                <SelectItem value="closed">Filled Orders</SelectItem>
                <SelectItem value="all">All Orders</SelectItem>
              </SelectContent>
            </Select>
            
            {statusFilter === 'open' && (
              <Button 
                variant="destructive" 
                size="sm" 
                onClick={handleCancelAllOrders}
                disabled={orders.length === 0 || isLoading}
              >
                Cancel All
              </Button>
            )}
          </div>
        </CardHeader>
        <CardContent>
          {error ? (
            <div className="flex items-center justify-center text-destructive p-4">
              {/* <AlertCircle className="h-4 w-4 mr-2" /> */}
              {error}
            </div>
          ) : isLoading ? (
            <div className="text-center p-4">Loading orders...</div>
          ) : orders.length === 0 ? (
            <div className="text-center text-muted-foreground p-4">
              No {statusFilter === 'all' ? '' : statusFilter} orders found.
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Symbol</TableHead>
                  <TableHead>Side</TableHead>
                  <TableHead>Type</TableHead>
                  <TableHead>Qty</TableHead>
                  <TableHead>Filled</TableHead>
                  <TableHead>Price</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Created</TableHead>
                  <TableHead>Updated</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {orders.map((order) => (
                  <TableRow key={order.id}>
                    <TableCell className="font-medium">{order.symbol}</TableCell>
                    <TableCell className={order.side === 'buy' ? 'text-green-500' : 'text-red-500'}>
                      {order.side.toUpperCase()}
                    </TableCell>
                    <TableCell>{order.type}</TableCell>
                    <TableCell>{order.qty}</TableCell>
                    <TableCell>{order.filled_qty}</TableCell>
                    <TableCell>
                      {order.filled_avg_price || order.limit_price || order.stop_price || 'Market'}
                    </TableCell>
                    <TableCell>{getStatusBadge(order.status)}</TableCell>
                    <TableCell>{formatDate(order.created_at)}</TableCell>
                    <TableCell>{formatDate(order.updated_at)}</TableCell>
                    <TableCell className="text-right">
                      {['new', 'partially_filled', 'pending_new'].includes(order.status) && (
                        <Button 
                          variant="ghost" 
                          size="sm" 
                          onClick={() => handleCancelOrder(order.id)}
                          className="text-destructive"
                        >
                          Cancel
                        </Button>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  );
}