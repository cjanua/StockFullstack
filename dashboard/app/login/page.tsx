'use client';

import { useState } from 'react';
import { useSearchParams } from 'next/navigation';
// import Link from 'next/link';
import { z } from 'zod';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import {
  Card,
  CardContent,
  CardDescription,
  // CardFooter,
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
} from '@/components/ui/form';
import { AlertCircle, Loader2 } from 'lucide-react';
import { toast } from '@/hooks/use-toast';
import { Alert, AlertDescription } from '@/components/ui/alert';

const formSchema = z.object({
  username: z.string().min(1, { message: 'Username is required' }),
  password: z.string().min(1, { message: 'Password is required' }),
});

export default function LoginPage() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // const router = useRouter();
  const searchParams = useSearchParams();
  const redirectUrl = searchParams!.get('redirect') || '/account';

  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      username: '',
      password: '',
    },
  });

  async function onSubmit(values: z.infer<typeof formSchema>) {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(values),
        credentials: 'include',
      });

      if (!response.ok) {
        let errorMessage = 'Login failed';
        // Try to parse error response
        try {
          const errorData = await response.json();
          errorMessage = errorData.error || errorMessage;
        } catch (parseError) {
          console.error('Failed to parse error response:', parseError);
          // If we can't parse the error, use status text
          errorMessage = `Server error: ${response.status} ${response.statusText}`;
        }
        
        throw new Error(errorMessage);
      }

      // Parse successful response
      let data;
      try {
        data = await response.json();
      } catch (parseError) {
        console.error('Failed to parse success response:', parseError);
        throw new Error('Invalid response from server');
      }

      form.reset();

      toast({
        title: 'Login Successful',
        description: `Welcome back, ${data.user.username}!`,
      });

      // Redirect to account page or the requested URL
      // router.push(decodeURIComponent(redirectUrl));
      window.location.href = decodeURIComponent(redirectUrl);
    } catch (err) {
      console.error('Login error:', err);
      let message = 'An error occurred';
      if (err instanceof Error) {
        message = err.message;
      } else if (err instanceof TypeError && err.message.includes('fetch')) {
        message = 'Network error - please check your connection';
      }

      setError(message);
      

      if (message.includes('Too many login attempts')) {
        form.setError('root', {
          type: 'manual',
          message: 'Account temporarily locked. Please try again in 15 minutes.'
        });
      }else if (message.includes('Network error')) {
        form.setError('root', {
          type: 'manual', 
          message: 'Cannot connect to server. Please check if the server is running.'
        });
      }
    } finally {
      setIsLoading(false);
    }
  }

return (
    <div className="min-h-screen flex items-center justify-center bg-background">
      <div className="w-full max-w-md px-4">
        <Card>
          <CardHeader className="space-y-1">
            <CardTitle className="text-2xl font-bold text-center">
              Sign In
            </CardTitle>
            <CardDescription className="text-center">
              Enter your credentials to access your trading account
            </CardDescription>
          </CardHeader>
          
          <CardContent>
            {error && (
              <Alert variant="destructive" className="mb-4">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            <Form {...form}>
              <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
                <FormField
                  control={form.control}
                  name="username"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Username</FormLabel>
                      <FormControl>
                        <Input 
                          placeholder="Enter username" 
                          autoComplete="username"
                          autoFocus
                          {...field} 
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                
                <FormField
                  control={form.control}
                  name="password"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Password</FormLabel>
                      <FormControl>
                        <Input 
                          type="password" 
                          placeholder="Enter password" 
                          autoComplete="current-password"
                          {...field} 
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                
                <Button 
                  type="submit" 
                  className="w-full"
                  disabled={isLoading}
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Signing in...
                    </>
                  ) : (
                    'Sign In'
                  )}
                </Button>
              </form>
            </Form>

            <div className="mt-4 text-center text-sm text-muted-foreground">
              <p>Access restricted to authorized users</p>
              {/* Temporary helper for forgotten password */}
              <p className="mt-2 text-xs">
                Contact admin if you need password reset
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Development helper - REMOVE IN PRODUCTION */}
        {process.env.NODE_ENV === 'development' && (
          <div className="mt-4 p-4 bg-muted rounded-lg text-xs">
            <p className="font-semibold mb-2">Dev Info:</p>
            <p>Security logs are in server console</p>
            <p>Check terminal for [SECURITY] logs</p>
            <p>Only user ID #1 can login (whitelist active)</p>
          </div>
        )}
      </div>
    </div>
  );
}