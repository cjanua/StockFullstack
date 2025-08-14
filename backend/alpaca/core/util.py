# alpaca/util.py

import logging
from logging import Logger as L
import json
import csv
from io import StringIO
from tabulate import tabulate

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')

class Logger(L):
    def __init__(self, name: str):
        super().__init__(name)
        self.addHandler(logging.StreamHandler())
        self.propagate = False

logger = Logger(__name__)

# Removed CommandContext dependency - define format_output with a generic parameter
def format_output(data: dict, ctx=None) -> str:
    """Format the output according to the specified format"""
    format_type = getattr(ctx, 'format', 'json') if ctx else 'json'
    pretty = getattr(ctx, 'pretty', False) if ctx else False

    if format_type == 'json':
        return json.dumps(data, indent=2) if pretty else json.dumps(data)
    elif format_type == 'csv':
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=data.keys() if isinstance(data, dict) else data[0].keys())
        writer.writeheader()
        writer.writerow(data) if isinstance(data, dict) else writer.writerows(data)
        return output.getvalue()
    elif format_type == 'table':
        if isinstance(data, dict):
            return tabulate(list(data.items()), headers=['Field', 'Value'])
        return tabulate(data, headers='keys')
    return str(data)
