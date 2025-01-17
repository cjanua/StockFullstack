from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class CommandContext:
    """Stores common parameters and configuration for commands"""
    debug: bool = False
    format: str = 'json'
    pretty: bool = False

class CommandRegistry:
    def __init__(self):
        self.commands: Dict[str, Dict[str, Dict[str, callable]]] = {}
    
    def register(self, domain: str, resource: str, action: str = 'get'):
        """Decorator to register command handlers"""
        def decorator(func):
            if domain not in self.commands:
                self.commands[domain] = {}
            if resource not in self.commands[domain]:
                self.commands[domain][resource] = {}
            self.commands[domain][resource][action] = func
            return func
        return decorator

    def get_handler(self, domain: str, resource: str, action: str = 'get') -> Optional[callable]:
        return self.commands.get(domain, {}).get(resource, {}).get(action)