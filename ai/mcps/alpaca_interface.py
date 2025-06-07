# MCP server setup for natural language trading
from mcp import Client

class MCPTradingInterface:
    def __init__(self, alpaca_credentials):
        self.mcp_client = Client("alpaca-mcp-server")
        self.setup_tools()
    
    async def natural_language_command(self, command):
        """Process natural language trading commands"""
        # Example: "Buy 100 shares of AAPL with 2% stop loss"
        response = await self.mcp_client.call_tool(
            "parse_trading_command",
            {"command": command}
        )
        return await self.execute_parsed_command(response)