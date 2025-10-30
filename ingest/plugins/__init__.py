from __future__ import annotations
import pkgutil
import inspect
from typing import Dict, Type
from .base import DataSource

def discover_plugins(module):
    """Discover all DataSource plugins in a given module."""
    plugins = {}
    for _, name, _ in pkgutil.iter_modules(module.__path__):
        __import__(f"{module.__name__}.{name}")
    
    for subclass in DataSource.__subclasses__():
        plugins[subclass.name] = subclass
    return plugins

# Discover and register all plugins in the 'ingest.plugins' module
PLUGIN_REGISTRY: Dict[str, Type[DataSource]] = discover_plugins(__import__(__name__, fromlist=[' ']))
