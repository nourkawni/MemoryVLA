# Coding Standards

These standards apply to any new code written in this project. Follow them for every new file and every function you add or modify.

## Code Comment Standard

### 1. File header
Every file must start with a block comment (before any import statements) describing:
- What the file does
- Its role in the system

### 2. Function comment block
Every function must contain a structured comment block with four parts:
- What the function does
- Return type and shape of the return value
- Example input
- Example output

### 3. Variable type comments
Every variable must be preceded by an inline comment stating its Python type (e.g. `# str`, `# list[dict]`, `# dict or None`).

## Example

```python
"""
data_loader.py

Loads and validates raw sensor recordings from disk and converts them
into normalized episode records used by the training pipeline.
"""

import json
import pathlib


def load_episode(path):
    """
    What it does:
        Reads a single episode JSON file from disk and parses it into
        a Python dict, validating that required keys are present.

    Returns:
        dict — parsed episode record with keys "id", "frames", "meta".

    Example input:
        load_episode("data/episode_001.json")

    Example output:
        {"id": "episode_001", "frames": [...], "meta": {"length": 120}}
    """
    file_path = pathlib.Path(path)  # pathlib.Path
    raw_text = file_path.read_text()  # str
    episode = json.loads(raw_text)  # dict

    required_keys = {"id", "frames", "meta"}  # set[str]
    missing = required_keys - episode.keys()  # set[str]
    if missing:
        raise ValueError(f"Missing keys: {missing}")

    return episode
```

## Notes
- Apply this to all newly written or substantially rewritten code, in every language used in this project (adapt "Python type" to the equivalent type annotation convention for non-Python code, e.g. TypeScript types, Go types).
- Do not retroactively rewrite untouched existing code just to add these comments — apply the standard going forward.
