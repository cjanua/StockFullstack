'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { z } from 'zod';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
  FormDescription,
} from '@/components/ui/form';
import {
  AlertCircle,
  AlertTriangle,
  CheckCircle,
  Loader2,
  RefreshCw,
  LogOut,
  Shield,
} from 'lucide-react';
import { Separator } from '@/components/ui/separator';
import { Switch } from '@/components/ui/switch';
import { toast } from '@/hooks/use-toast';
import {
  Alert,
  AlertDescription,
  AlertTitle,
} from '@/components/ui/alert';

interface UserProfile {
  id: number;
  username: string;
  email: string;
  created_at: string;
  alpaca: {
    connected: boolean;
    status: 'active' | 'inactive' | 'error';
    message: string;
    has_credentials: boolean;
    last_verified: string | null;
  };
}

const alpacaFormSchema = z.object({
  alpaca_key: z.string().min(1, { message: 'API Key is required' }),
  alpaca_secret: z.string().min(1, { message: 'API Secret is required' }),
  paper: z.boolean().default(false),
});

export default function AccountPage() {
  const [user, setUser] = useState<UserProfile | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isUpdating, setIsUpdating] = useState(false);
  const [isCheckingConnection, setIsCheckingConnection] = useState(false);
  const router = useRouter();

  const alpacaForm = useForm<z.infer<typeof alpacaFormSchema>>({
    resolver: zodResolver(alpacaFormSchema),
    defaultValues: {
      alpaca_key: '',
      alpaca_secret: '',
      paper: false,
    },
  });

  // Fetch user profile on page load
  useEffect(() => {
    async function fetchUserProfile() {
      try {
        const response = await fetch('/api/auth/me');
        
        if (!response.ok) {
          if (response.status === 401) {
            // Not authenticated, redirect to login
            router.push('/login');
            return;
          }
          throw new Error('Failed to fetch user profile');
        }
        
        const userData = await response.json();
        setUser(userData);
      } catch (error) {
        toast({
          title: 'Error',
          description: error instanceof Error ? error.message : 'An error occurred',
          variant: 'destructive',
        });
      } finally {
        setIsLoading(false);
      }
    }
    
    fetchUserProfile();
  }, [router]);

  // Handle Alpaca connection form submission
  async function onConnectAlpaca(values: z.infer<typeof alpacaFormSchema>) {
    setIsUpdating(true);
    
    try {
      const response = await fetch('/api/alpaca/connect', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(values),
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || 'Failed to connect to Alpaca');
      }
      
      toast({
        title: 'Success',
        description: 'Successfully connected to Alpaca API.',
      });
      
      // Refresh user data to show updated Alpaca status
      await refreshUserData();
    } catch (error) {
      toast({
        title: 'Connection Failed',
        description: error instanceof Error ? error.message : 'An error occurred',
        variant: 'destructive',
      });
    } finally {
      setIsUpdating(false);
    }
  }

  // Handle refresh connection button click
  async function handleRefreshConnection() {
    setIsCheckingConnection(true);
    
    try {
      await refreshUserData();
      
      toast({
        title: 'Connection Refreshed',
        description: 'Successfully refreshed Alpaca connection status.',
      });
    } catch (error) {
      toast({
        title: 'Refresh Failed',
        description: error instanceof Error ? error.message : 'An error occurred',
        variant: 'destructive',
      });
    } finally {
      setIsCheckingConnection(false);
    }
  }

  // Helper function to refresh user data
  async function refreshUserData() {
    const response = await fetch('/api/auth/me');
    if (!response.ok) {
      throw new Error('Failed to refresh user data');
    }
    const userData = await response.json();
    setUser(userData);
  }

  // Handle logout
  async function handleLogout() {
    try {
      await fetch('/api/auth/logout', { method: 'POST' });
      router.push('/login');
    } catch (error) {
      toast({
        title: 'Logout Failed',
        description: error instanceof Error ? error.message : 'An error occurred',
        variant: 'destructive',
      });
    }
  }

  if (isLoading) {
    return (
      <div className="flex justify-center items-center min-h-screen">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  return (
    <div className="container mx-auto py-10 px-4">
      <div className="flex justify-between items-center mb-8">
        <h1 className="text-3xl font-bold">Account Settings</h1>
        <Button variant="outline" onClick={handleLogout}>
          <LogOut className="mr-2 h-4 w-4" />
          Logout
        </Button>
      </div>

      {user && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* User Profile Card */}
            <Card>
              <CardHeader>
                <CardTitle>Profile Information</CardTitle>
                <CardDescription>Your account details</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <p className="text-sm font-medium">Username</p>
                  <p className="text-lg">{user.username}</p>
                </div>
                <div>
                  <p className="text-sm font-medium">Email</p>
                  <p className="text-lg">{user.email}</p>
                </div>
                <div>
                  <p className="text-sm font-medium">Account Created</p>
                  <p className="text-lg">
                    {new Date(user.created_at).toLocaleDateString()}
                  </p>
                </div>
              </CardContent>
            </Card>

            {/* Alpaca Connection Status Card */}
            <Card className="md:col-span-2">
              <CardHeader>
                <div className="flex justify-between items-center">
                  <div>
                    <CardTitle>Alpaca Connection</CardTitle>
                    <CardDescription>Your trading API connection status</CardDescription>
                  </div>
                  
                  {user.alpaca.has_credentials && (
                    <Button 
                      variant="outline" 
                      size="sm"
                      onClick={handleRefreshConnection}
                      disabled={isCheckingConnection}
                    >
                      {isCheckingConnection ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <RefreshCw className="h-4 w-4" />
                      )}
                      <span className="ml-2">Refresh</span>
                    </Button>
                  )}
                </div>
              </CardHeader>
              <CardContent>
                <div className="mb-4">
                  {user.alpaca.status === 'active' ? (
                    <Alert className="bg-green-50 dark:bg-green-950 border-green-200">
                      <CheckCircle className="h-4 w-4 text-green-500" />
                      <AlertTitle className="text-green-700 dark:text-green-300">Connected</AlertTitle>
                      <AlertDescription className="text-green-600 dark:text-green-400">
                        {user.alpaca.message}
                      </AlertDescription>
                    </Alert>
                  ) : user.alpaca.status === 'error' ? (
                    <Alert variant="destructive">
                      <AlertCircle className="h-4 w-4" />
                      <AlertTitle>Connection Error</AlertTitle>
                      <AlertDescription>
                        {user.alpaca.message}
                      </AlertDescription>
                    </Alert>
                  ) : (
                    <Alert variant="default" className="bg-amber-50 dark:bg-amber-950 border-amber-200">
                      <AlertTriangle className="h-4 w-4 text-amber-500" />
                      <AlertTitle className="text-amber-700 dark:text-amber-300">Not Connected</AlertTitle>
                      <AlertDescription className="text-amber-600 dark:text-amber-400">
                        {user.alpaca.message}
                      </AlertDescription>
                    </Alert>
                  )}
                </div>

                {user.alpaca.last_verified && (
                  <p className="text-sm text-muted-foreground mt-2">
                    Last verified: {new Date(user.alpaca.last_verified).toLocaleString()}
                  </p>
                )}
                
                <Separator className="my-6" />
                
                <Form {...alpacaForm}>
                  <form onSubmit={alpacaForm.handleSubmit(onConnectAlpaca)} className="space-y-4">
                    <FormField
                      control={alpacaForm.control}
                      name="alpaca_key"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Alpaca API Key</FormLabel>
                          <FormControl>
                            <Input 
                              placeholder="Enter your Alpaca API Key"
                              autoComplete="off"
                              {...field} 
                            />
                          </FormControl>
                          <FormDescription>
                            Your API key from the Alpaca dashboard
                          </FormDescription>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                    
                    <FormField
                      control={alpacaForm.control}
                      name="alpaca_secret"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Alpaca API Secret</FormLabel>
                          <FormControl>
                            <Input 
                              type="password" 
                              placeholder="Enter your Alpaca API Secret" 
                              autoComplete="new-password"
                              {...field} 
                            />
                          </FormControl>
                          <FormDescription>
                            Your API secret from the Alpaca dashboard
                          </FormDescription>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                    
                    <FormField
                      control={alpacaForm.control}
                      name="paper"
                      render={({ field }) => (
                        <FormItem className="flex flex-row items-center justify-between rounded-lg border p-4">
                          <div className="space-y-0.5">
                            <FormLabel className="text-base">
                              Paper Trading
                            </FormLabel>
                            <FormDescription>
                              Use paper trading account (practice with virtual money)
                            </FormDescription>
                          </div>
                          <FormControl>
                            <Switch
                              checked={field.value}
                              onCheckedChange={field.onChange}
                            />
                          </FormControl>
                        </FormItem>
                      )}
                    />
                    
                    <Button 
                      type="submit" 
                      className="w-full"
                      disabled={isUpdating}
                    >
                      {isUpdating ? (
                        <>
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                          Connecting...
                        </>
                      ) : (
                        <>
                          <Shield className="mr-2 h-4 w-4" />
                          {user.alpaca.has_credentials ? 'Update Connection' : 'Connect to Alpaca'}
                        </>
                      )}
                    </Button>
                  </form>
                </Form>
              </CardContent>
            </Card>
          </div>
        </>
      )}
    </div>
  );
}