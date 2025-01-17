# Configure logging
import logging
from logging import Logger as L

from backend.alpaca.commands import CommandContext

logging.basicConfig(
    # level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class Logger(L):
    def __init__(self, name: str):
        super().__init__(name)
        self.setLevel(logging.DEBUG)
        self.addHandler(logging.StreamHandler())
        self.propagate = False
        

logger = Logger(__name__)


def format_output(data: dict, ctx: CommandContext) -> str:
    """Format the output according to the specified format"""
    import json
    from tabulate import tabulate
    import csv
    from io import StringIO
    
    if ctx.format == 'json':
        if ctx.pretty:
            return json.dumps(data, indent=2)
        return json.dumps(data)
    
    elif ctx.format == 'csv':
        output = StringIO()
        if isinstance(data, dict):
            writer = csv.DictWriter(output, fieldnames=data.keys())
            writer.writeheader()
            writer.writerow(data)
        elif isinstance(data, list):
            writer = csv.DictWriter(output, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        return output.getvalue()
    
    elif ctx.format == 'table':
        if isinstance(data, dict):
            return tabulate([(k, v) for k, v in data.items()], headers=['Field', 'Value'])
        elif isinstance(data, list):
            return tabulate(data, headers='keys')
    
    return str(data)
