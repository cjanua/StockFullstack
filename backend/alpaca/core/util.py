# alpaca/util.py

import logging
from logging import Logger as L
from backend.alpaca.cli.commands import CommandContext
from tabulate import tabulate
import json
import csv
from io import StringIO

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')

class Logger(L):
    def __init__(self, name: str):
        super().__init__(name)
        self.addHandler(logging.StreamHandler())
        self.propagate = False

logger = Logger(__name__)

def format_output(data: dict, ctx: CommandContext) -> str:
    """Format the output according to the specified format"""
    if ctx.format == 'json':
        return json.dumps(data, indent=2) if ctx.pretty else json.dumps(data)
    elif ctx.format == 'csv':
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=data.keys() if isinstance(data, dict) else data[0].keys())
        writer.writeheader()
        writer.writerow(data) if isinstance(data, dict) else writer.writerows(data)
        return output.getvalue()
    elif ctx.format == 'table':
        return tabulate([(k, v) for k, v in data.items()], headers=['Field', 'Value']) if isinstance(data, dict) else tabulate(data, headers='keys')
    return str(data)
