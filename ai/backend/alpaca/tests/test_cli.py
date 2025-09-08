import unittest
import subprocess
import json


PYTHON='/home/cjanua/Documents/repos/Stock-FullStack/StockFullstack/venv/bin/python3.13'

class TestCLI(unittest.TestCase):
    def run_cli_command(self, command):
        """Helper function to run CLI commands and return the output."""
        result = subprocess.run(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return result

    def test_get_account(self):
        """Test the trading/account command."""
        command = f"{PYTHON} /home/cjanua/Documents/repos/Stock-FullStack/StockFullstack/backend/alpaca/cli/apca.py trading/account --format json"
        result = self.run_cli_command(command)
        self.assertEqual(result.returncode, 0, f"Command failed: {result.stderr}")
        output = json.loads(result.stdout)
        self.assertIn("id", output, "Account ID not found in response")
        self.assertIn("portfolio_value", output, "Portfolio value not found in response")

    def test_get_positions(self):
        """Test the trading/account/positions command."""
        command = f"{PYTHON} /home/cjanua/Documents/repos/Stock-FullStack/StockFullstack/backend/alpaca/cli/apca.py trading/account/positions --format json"
        result = self.run_cli_command(command)
        self.assertEqual(result.returncode, 0, f"Command failed: {result.stderr}")
        output = json.loads(result.stdout)
        self.assertIsInstance(output, list, "Positions response should be a list")
        if output:
            self.assertIn("symbol", output[0], "Symbol not found in position response")

    def test_search_assets(self):
        """Test the trading/assets/search command."""
        command = f"{PYTHON} /home/cjanua/Documents/repos/Stock-FullStack/StockFullstack/backend/alpaca/cli/apca.py trading/assets/search --query AAPL --format json"
        result = self.run_cli_command(command)
        self.assertEqual(result.returncode, 0, f"Command failed: {result.stderr}")
        output = json.loads(result.stdout)
        self.assertIsInstance(output, list, "Search response should be a list")
        self.assertTrue(
            any(asset.get("symbol") == "AAPL" for asset in output),
            "Symbol 'AAPL' not found in search response"
        )

if __name__ == "__main__":
    unittest.main()
