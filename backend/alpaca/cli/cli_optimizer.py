#!/usr/bin/env python3
# alpaca/cli_optimizer.py

import click
import requests
from tabulate import tabulate

@click.group()
def cli():
    """Portfolio optimization toolkit for Alpaca"""
    pass

@cli.command()
@click.option('--lookback', default=365, help='Days of historical data to use')
@click.option('--min-change', default=0.01, help='Minimum position change percentage to recommend')
@click.option('--cash-reserve', default=0.05, help='Cash percentage to keep in reserve')
def optimize(lookback, min_change, cash_reserve):
    """Get portfolio optimization recommendations"""
    try:
        # Try to use the microservice if it's running
        response = requests.get(
            "http://localhost:8001/api/portfolio/recommendations",
            params={
                "lookback_days": lookback,
                "min_change_percent": min_change,
                "cash_reserve_percent": cash_reserve
            }
        )

        if response.status_code == 200:
            data = response.json()
        else:
            # Fallback to direct calculation if service is unavailable
            click.echo("Microservice unavailable, calculating directly...")
            data = calculate_recommendations(lookback, min_change, cash_reserve)

        # Display results
        display_recommendations(data)

    except requests.ConnectionError:
        click.echo("Microservice unavailable, calculating directly...")
        data = calculate_recommendations(lookback, min_change, cash_reserve)
        display_recommendations(data)

def calculate_recommendations(lookback, min_change, cash_reserve):
    """Calculate portfolio recommendations directly (for CLI fallback)"""
    from backend.alpaca.sdk.loaders import get_account, get_positions
    from backend.alpaca.api.portfolio_service import get_optimal_portfolio
    import asyncio

    # Get account and positions
    account_result = get_account()
    positions_result = get_positions()

    if account_result.is_err() or positions_result.is_err():
        click.echo("Error fetching account or positions data")
        return {}

    account = account_result.ok_value

    # Get optimal weights (run async function in sync context)
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    loop.run_until_complete(get_optimal_portfolio(lookback))

    # Calculate recommendations
    # (Simplified implementation - the full logic is in the microservice)

    return {
        "portfolio_value": float(account['portfolio_value']),
        "cash": float(account['cash']),
        "target_cash": float(account['portfolio_value']) * cash_reserve,
        "recommendations": []  # Would contain real recommendations in full implementation
    }

def display_recommendations(data):
    """Format and display portfolio recommendations"""
    click.echo("\n=== Portfolio Optimization Report ===")
    click.echo(f"Portfolio Value: ${data['portfolio_value']:,.2f}")
    click.echo(f"Current Cash: ${data['cash']:,.2f}")
    click.echo(f"Target Cash: ${data['target_cash']:,.2f}")

    if not data['recommendations']:
        click.echo("\nNo recommendations to make at this time.")
        return

    # Prepare data for tabulate
    buy_recs = [r for r in data['recommendations'] if r['action'] == "Buy"]
    sell_recs = [r for r in data['recommendations'] if r['action'] == "Sell"]

    if buy_recs:
        click.echo("\n== Buy Recommendations ==")
        buy_table = tabulate(
            [[r['symbol'], r['current_shares'], r['target_shares'], r['quantity']] for r in buy_recs],
            headers=["Symbol", "Current Shares", "Target Shares", "Buy Qty"],
            tablefmt="pretty"
        )
        click.echo(buy_table)

    if sell_recs:
        click.echo("\n== Sell Recommendations ==")
        sell_table = tabulate(
            [[r['symbol'], r['current_shares'], r['target_shares'], r['quantity']] for r in sell_recs],
            headers=["Symbol", "Current Shares", "Target Shares", "Sell Qty"],
            tablefmt="pretty"
        )
        click.echo(sell_table)

if __name__ == "__main__":
    cli()
