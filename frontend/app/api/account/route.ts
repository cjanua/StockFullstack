import { getAlpacaAccount } from '@/lib/alpaca'
import { NextResponse } from 'next/server'


export async function GET() {
  try {
    const account = await getAlpacaAccount()
    return NextResponse.json(account)
  } catch (error) {
    console.error('Error fetching account:', error)
    return NextResponse.json(
      { error: 'Failed to fetch account data' },
      { status: 500 }
    )
  }
}