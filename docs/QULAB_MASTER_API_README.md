# QuLabInfinite Master API

**Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.**

## Overview

The QuLabInfinite Master API provides a comprehensive, unified interface to access all 80+ specialized scientific laboratories in the QuLabInfinite ecosystem. It handles import errors gracefully, categorizes labs by scientific domain, provides powerful search capabilities, and offers a production-ready API for lab discovery and execution.

## Features

- **Unified Interface**: Single API to access 80+ specialized scientific laboratories
- **Domain Categorization**: Automatic categorization by Physics, Chemistry, Biology, Engineering, Medicine, Earth Science, Computer Science, Mathematics, Materials Science, and Quantum Science
- **Graceful Error Handling**: Continues loading even when some labs have missing dependencies
- **Powerful Search**: Search labs by name, description, keywords, or domain
- **Capability Inspection**: Discover what methods each lab provides
- **Demo Execution**: Run demonstrations to test lab functionality
- **Production Ready**: Comprehensive logging, error handling, and documentation

## Quick Start

### Basic Usage

```python
from qulab_master_api import QuLabMasterAPI

# Initialize the API
api = QuLabMasterAPI(auto_load=True, verbose=True)

# Get statistics
stats = api.get_statistics()
print(f"Total Labs: {stats['total_labs']}")
print(f"Available: {stats['available_labs']}")

# List all available labs
labs = api.list_labs(available_only=True)
for lab in labs:
    print(f"- {lab['display_name']}: {lab['description']}")
```

### Search for Labs

```python
# Search for quantum-related labs
results = api.search_labs("quantum")
for result in results:
    print(f"{result['display_name']} (Relevance: {result['relevance_score']})")

# Search by domain
from qulab_master_api import LabDomain
physics_labs = api.list_labs(domain=LabDomain.PHYSICS, available_only=True)
```

### Get and Use a Lab

```python
# Get a specific lab instance
materials_lab = api.get_lab("materials_lab")

if materials_lab:
    # Use the lab
    result = materials_lab.optimize_material_properties(...)
    print(result)

# Get lab capabilities
capabilities = api.get_capabilities("materials_lab")
print(f"Available methods: {capabilities['capabilities']}")
```

### Run Lab Demonstrations

```python
# Run a demo
result = api.run_demo("thermodynamics")
if result.get("success"):
    print("Demo successful!")
    print(result["result"])
```

## Command-Line Interface

The Master API includes a powerful CLI for exploration:

### List All Labs

```bash
python qulab_master_api.py list
```

### List Labs by Domain

```bash
python qulab_master_api.py list --domain Physics --available-only
```

### Search for Labs

```bash
python qulab_master_api.py search quantum
python qulab_master_api.py search "molecular biology"
```

### Get Lab Information

```bash
python qulab_master_api.py info materials_lab
```

### Run Lab Demo

```bash
python qulab_master_api.py demo thermodynamics
```

### Get Statistics

```bash
python qulab_master_api.py stats
```

### Export Catalog

```bash
python qulab_master_api.py export --output catalog.json
```

## Available Labs by Domain

### Physics (13 labs)
- Quantum Mechanics Lab
- Quantum Computing Lab
- Advanced Quantum Lab
- Particle Physics Lab
- Nuclear Physics Lab
- Condensed Matter Physics Lab
- Plasma Physics Lab
- Astrophysics Lab
- Fluid Dynamics Lab
- Thermodynamics Lab
- Electromagnetism Lab
- Optics & Photonics Lab
- Frequency Lab

### Chemistry (11 labs)
- Chemistry Lab
- Organic Chemistry Lab
- Inorganic Chemistry Lab
- Physical Chemistry Lab
- Analytical Chemistry Lab
- Computational Chemistry Lab
- Polymer Chemistry Lab
- Biochemistry Lab
- Electrochemistry Lab
- Catalysis Lab
- Atmospheric Chemistry Lab

### Biology (18 labs)
- Molecular Biology Lab
- Cell Biology Lab
- Genetics Lab
- Genomics Lab
- Bioinformatics Lab
- Microbiology Lab
- Immunology Lab
- Neuroscience Lab
- Developmental Biology Lab
- Evolutionary Biology Lab
- Ecology Lab
- Astrobiology Lab
- Proteomics Lab
- Protein Folding Lab
- Protein Engineering Lab
- Biological Quantum Lab

### Medicine (9 labs)
- Oncology Lab
- Realistic Tumor Lab
- Cardiology Lab
- Cardiovascular Plaque Lab
- Cardiac Fibrosis Lab
- Neurology Lab
- Drug Design Lab
- Pharmacology Lab
- Toxicology Lab
- Medical Imaging Lab
- Clinical Trials Simulation Lab
- Drug Interaction Simulator Lab

### Engineering (9 labs)
- Chemical Engineering Lab
- Biomedical Engineering Lab
- Mechanical Engineering Lab
- Electrical Engineering Lab
- Structural Engineering Lab
- Aerospace Engineering Lab
- Environmental Engineering Lab
- Robotics Lab
- Control Systems Lab

### Earth Science (10 labs)
- Geology Lab
- Seismology Lab
- Geophysics Lab
- Meteorology Lab
- Atmospheric Science Lab
- Oceanography Lab
- Hydrology Lab
- Climate Modeling Lab
- Renewable Energy Lab
- Carbon Capture Lab

### Computer Science (7 labs)
- Machine Learning Lab
- Deep Learning Lab
- Neural Networks Lab
- Natural Language Processing Lab
- Computer Vision Lab
- Cryptography Lab
- Signal Processing Lab

### Mathematics (3 labs)
- Algorithm Design Lab
- Graph Theory Lab
- Optimization Theory Lab

### Materials Science (3 labs)
- Materials Science Lab
- Advanced Materials Lab
- Materials Chemistry Lab

### Quantum Science (3 labs)
- Quantum Mechanics Lab
- Quantum Computing Lab
- Biological Quantum Lab

## API Reference

### QuLabMasterAPI Class

#### `__init__(auto_load=True, verbose=True)`
Initialize the Master API.

**Parameters:**
- `auto_load` (bool): Automatically load all labs on initialization
- `verbose` (bool): Enable verbose logging

#### `list_labs(domain=None, available_only=False)`
List all laboratories.

**Parameters:**
- `domain` (LabDomain, optional): Filter by specific domain
- `available_only` (bool): Only return successfully loaded labs

**Returns:**
- List of lab information dictionaries

#### `get_lab(lab_name)`
Get a specific laboratory instance.

**Parameters:**
- `lab_name` (str): Name of the lab

**Returns:**
- Lab instance if available, None otherwise

#### `search_labs(query)`
Search for labs by keyword or description.

**Parameters:**
- `query` (str): Search query string

**Returns:**
- List of matching lab information with relevance scores

#### `get_capabilities(lab_name)`
Get capabilities of a specific lab.

**Parameters:**
- `lab_name` (str): Name of the lab

**Returns:**
- Dictionary with lab capabilities and metadata

#### `run_demo(lab_name, **kwargs)`
Run a demonstration of a specific lab.

**Parameters:**
- `lab_name` (str): Name of the lab
- `**kwargs`: Additional arguments for the demo

**Returns:**
- Dictionary with demo results

#### `get_statistics()`
Get statistics about loaded labs.

**Returns:**
- Dictionary with comprehensive statistics

#### `export_catalog(output_path=None)`
Export complete lab catalog to JSON.

**Parameters:**
- `output_path` (str, optional): Path to save JSON file

**Returns:**
- JSON string of catalog

### LabDomain Enum

Available domains for filtering:
- `LabDomain.PHYSICS`
- `LabDomain.CHEMISTRY`
- `LabDomain.BIOLOGY`
- `LabDomain.ENGINEERING`
- `LabDomain.MEDICINE`
- `LabDomain.EARTH_SCIENCE`
- `LabDomain.COMPUTER_SCIENCE`
- `LabDomain.MATHEMATICS`
- `LabDomain.MATERIALS`
- `LabDomain.QUANTUM`

## Examples

See `QULAB_MASTER_API_EXAMPLES.py` for comprehensive examples including:

1. Basic initialization and statistics
2. Listing labs with filters
3. Searching for labs
4. Inspecting lab capabilities
5. Using lab instances
6. Running demonstrations
7. Domain analysis
8. Exporting catalogs
9. Custom workflows
10. Batch processing
11. Programmatic lab selection
12. Error handling

Run all examples:
```bash
python QULAB_MASTER_API_EXAMPLES.py
```

## Error Handling

The Master API handles errors gracefully:

- **Import Errors**: Labs with missing dependencies are registered but marked as unavailable
- **Instantiation Errors**: Labs that fail to instantiate are logged but don't stop the loading process
- **Missing Labs**: Attempting to access non-existent labs returns None with appropriate error messages
- **Demo Failures**: Failed demos return error information without crashing

### Example Error Handling

```python
# Try to get a lab
lab = api.get_lab("my_lab")
if lab is None:
    print("Lab not available")

# Check capabilities first
caps = api.get_capabilities("my_lab")
if not caps['available']:
    print(f"Lab unavailable: {caps['error']}")
else:
    lab = api.get_lab("my_lab")
    # Use lab safely
```

## Performance

- **Fast Loading**: Labs are loaded on-demand when accessed
- **Caching**: Lab instances are cached after first load
- **Memory Efficient**: Only requested labs are instantiated
- **Scalable**: Handles 80+ labs without performance issues

## Integration Examples

### Flask Web API

```python
from flask import Flask, jsonify
from qulab_master_api import QuLabMasterAPI

app = Flask(__name__)
api = QuLabMasterAPI()

@app.route('/labs')
def list_labs():
    return jsonify(api.list_labs(available_only=True))

@app.route('/labs/<name>')
def get_lab(name):
    return jsonify(api.get_capabilities(name))

@app.route('/labs/<name>/demo', methods=['POST'])
def run_demo(name):
    result = api.run_demo(name)
    return jsonify(result)
```

### Jupyter Notebook

```python
from qulab_master_api import QuLabMasterAPI
import pandas as pd

# Initialize
api = QuLabMasterAPI()

# Create DataFrame of all labs
labs = api.list_labs()
df = pd.DataFrame(labs)

# Display statistics
df.groupby('domain')['available'].agg(['count', 'sum'])

# Search and filter
quantum_labs = df[df['name'].str.contains('quantum')]
```

### Async Usage

```python
import asyncio
from qulab_master_api import QuLabMasterAPI

async def process_labs():
    api = QuLabMasterAPI()

    # Get all available labs
    labs = api.list_labs(available_only=True)

    # Process in parallel
    tasks = []
    for lab_info in labs[:5]:
        task = asyncio.create_task(run_lab_demo(api, lab_info['name']))
        tasks.append(task)

    results = await asyncio.gather(*tasks)
    return results

async def run_lab_demo(api, lab_name):
    # Run in executor to not block
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, api.run_demo, lab_name)
```

## Troubleshooting

### Lab Not Loading

If a lab shows as unavailable:

1. Check error message:
   ```python
   caps = api.get_capabilities("lab_name")
   print(caps['error'])
   ```

2. Verify dependencies are installed
3. Check lab file exists and is correctly named
4. Ensure lab class name matches registry

### Import Errors

If you see import errors:
- Most labs require NumPy, SciPy, and other scientific Python packages
- Some labs have specific dependencies (PyTorch, TensorFlow, etc.)
- Install missing packages: `pip install numpy scipy torch`

### Performance Issues

If loading is slow:
- Use `auto_load=False` and load labs on-demand
- Filter by domain to reduce loaded labs
- Disable verbose logging: `verbose=False`

## Contributing

To add a new lab to the registry:

1. Create your lab file following the existing pattern
2. Add entry to `_build_lab_registry()` in `qulab_master_api.py`
3. Specify: name, display_name, description, domain, module, class, keywords
4. Test loading: `python qulab_master_api.py info your_lab_name`

## License

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

## Support

For questions, issues, or feature requests, please visit:
- Website: https://aios.is
- Red Team Tools: https://red-team-tools.aios.is
- GitHub: https://github.com/thegavl

---

**Built with QuLabInfinite - Advancing Scientific Discovery Through Unified Laboratory Access**
