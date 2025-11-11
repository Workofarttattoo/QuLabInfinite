# QuLab Master API - Quick Reference Card

**Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.**

## Import and Initialize

```python
from qulab_master_api import QuLabMasterAPI, LabDomain

api = QuLabMasterAPI(auto_load=True, verbose=False)
```

## Core Operations

### List Labs
```python
# All labs
labs = api.list_labs()

# Available only
labs = api.list_labs(available_only=True)

# By domain
labs = api.list_labs(domain=LabDomain.PHYSICS)
```

### Search
```python
# Search by keyword
results = api.search_labs("quantum")

# Access results
for lab in results:
    print(f"{lab['display_name']}: {lab['relevance_score']}")
```

### Get Lab Instance
```python
# Get lab
lab = api.get_lab("materials_lab")

# Check if available
if lab:
    result = lab.some_method()
```

### Get Capabilities
```python
# Get lab info
caps = api.get_capabilities("materials_lab")

# Check availability
if caps['available']:
    print(caps['capabilities'])
else:
    print(caps['error'])
```

### Run Demo
```python
# Run demo
result = api.run_demo("thermodynamics")

# Check success
if result.get('success'):
    print(result['result'])
```

### Statistics
```python
stats = api.get_statistics()
print(f"Total: {stats['total_labs']}")
print(f"Available: {stats['available_labs']}")
```

### Export Catalog
```python
# To file
api.export_catalog("catalog.json")

# Get JSON string
json_str = api.export_catalog()
```

## Command-Line

```bash
# List all labs
python qulab_master_api.py list

# List by domain
python qulab_master_api.py list --domain Physics --available-only

# Search
python qulab_master_api.py search quantum

# Get info
python qulab_master_api.py info materials_lab

# Run demo
python qulab_master_api.py demo thermodynamics

# Statistics
python qulab_master_api.py stats

# Export
python qulab_master_api.py export --output catalog.json
```

## Domains

```python
LabDomain.PHYSICS           # Physics labs
LabDomain.CHEMISTRY         # Chemistry labs
LabDomain.BIOLOGY           # Biology labs
LabDomain.ENGINEERING       # Engineering labs
LabDomain.MEDICINE          # Medicine labs
LabDomain.EARTH_SCIENCE     # Earth science labs
LabDomain.COMPUTER_SCIENCE  # CS labs
LabDomain.MATHEMATICS       # Math labs
LabDomain.MATERIALS         # Materials science labs
LabDomain.QUANTUM           # Quantum science labs
```

## Error Handling

```python
# Check lab availability
lab = api.get_lab("my_lab")
if lab is None:
    print("Lab not available")

# Check capabilities first
caps = api.get_capabilities("my_lab")
if not caps['available']:
    print(f"Error: {caps['error']}")

# Handle demo failures
result = api.run_demo("my_lab")
if 'error' in result:
    print(f"Demo failed: {result['error']}")
```

## Common Patterns

### Find Best Match
```python
results = api.search_labs("molecular")
best = results[0] if results else None
if best and best['available']:
    lab = api.get_lab(best['name'])
```

### Process All Available
```python
labs = api.list_labs(available_only=True)
for lab_info in labs:
    lab = api.get_lab(lab_info['name'])
    # Use lab
```

### Domain Analysis
```python
stats = api.get_statistics()
for domain, counts in stats['by_domain'].items():
    print(f"{domain}: {counts['available']}/{counts['total']}")
```

## Key Methods Summary

| Method | Purpose | Returns |
|--------|---------|---------|
| `list_labs()` | List all labs | List[Dict] |
| `get_lab()` | Get lab instance | Lab or None |
| `search_labs()` | Search by keyword | List[Dict] |
| `get_capabilities()` | Get lab info | Dict |
| `run_demo()` | Run lab demo | Dict |
| `get_statistics()` | Get stats | Dict |
| `export_catalog()` | Export to JSON | str |

## Files Location

```
/Users/noone/QuLabInfinite/
├── qulab_master_api.py              # Main API (1,461 lines)
├── QULAB_MASTER_API_EXAMPLES.py     # Examples (364 lines)
├── QULAB_MASTER_API_README.md       # Full docs (501 lines)
├── QULAB_MASTER_API_SUMMARY.md      # Implementation summary
└── QULAB_API_QUICK_REFERENCE.md     # This file
```

## Quick Test

```bash
# Run examples
python QULAB_MASTER_API_EXAMPLES.py

# Get stats
python qulab_master_api.py stats

# Search quantum
python qulab_master_api.py search quantum
```

## Contact

- Website: https://aios.is
- Email: echo@aios.is
- GitHub: https://github.com/thegavl

---

**80+ Labs. One API. Infinite Possibilities.**
