import hashlib
import inspect
import joblib
import os
from pathlib import Path
from functools import wraps
import pandas as pd

CACHE_DIR = Path(".cache/comprehensive")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def get_file_hash(path):
    """Computes the SHA256 hash of a file's content."""
    if not os.path.exists(path):
        return ""
    hasher = hashlib.sha256()
    with open(path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def get_func_hash(func):
    """Computes the SHA256 hash of a function's source code."""
    try:
        source = inspect.getsource(func)
        return hashlib.sha256(source.encode()).hexdigest()
    except (TypeError, OSError):
        # Fallback for built-in functions or functions defined in REPL
        return func.__name__

def generate_cache_key(func, dependencies=None, *args, **kwargs):
    """
    Generates a cache key based on function source, arguments, and file dependencies.
    """
    hasher = hashlib.sha256()

    # 1. Hash function's source code
    func_hash = get_func_hash(func)
    hasher.update(func_hash.encode())

    # 2. Hash file dependencies
    if dependencies:
        for dep in sorted(dependencies):
            file_hash = get_file_hash(dep)
            hasher.update(file_hash.encode())

    # 3. Hash args and kwargs
    for arg in args:
        if isinstance(arg, pd.DataFrame):
             hasher.update(joblib.hash(arg).encode())
        else:
             hasher.update(str(arg).encode())

    for key, value in sorted(kwargs.items()):
        hasher.update(key.encode())
        if isinstance(value, pd.DataFrame):
             hasher.update(joblib.hash(value).encode())
        else:
             hasher.update(str(value).encode())


    return f"{func.__name__}_{hasher.hexdigest()[:16]}.joblib"

def cache_on_disk(dependencies=None):
    """
    A decorator to cache function results on disk.
    `dependencies` is a list of file paths that the function depends on.
    If any of these files change, the cache is invalidated.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Re-generate key to handle cases where DataFrame is passed by keyword
            cache_key = generate_cache_key(func, dependencies, *args, **kwargs)
            cache_file = CACHE_DIR / cache_key

            if cache_file.exists():
                print(f"CACHE HIT: Loading result for {func.__name__} from {cache_file}")
                try:
                    return joblib.load(cache_file)
                except Exception as e:
                    print(f"CACHE ERROR: Failed to load {cache_file}. Recalculating. Error: {e}")


            print(f"CACHE MISS: Running {func.__name__} and caching result.")
            result = func(*args, **kwargs)
            joblib.dump(result, cache_file)
            return result
        return wrapper
    return decorator
