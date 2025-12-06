# Materials Project Matching Guide

This guide explains how to match your lab materials to Materials Project IDs.

## Quick Start

### 1. Set up your API key

Get a free API key from Materials Project:
1. Visit https://materialsproject.org/api
2. Sign up for a free account
3. Copy your API key
4. Set it as an environment variable:

```bash
export MP_API_KEY='your-api-key-here'
```

### 2. Test with a few materials first

Before processing all materials, test with a small batch:

```bash
# Test with first 10 materials
python3 scripts/match_materials_to_mp.py --limit 10 --dry-run
```

This will show you what matches would be made without modifying your database.

### 3. Run the full matching

Once you're satisfied with the test results:

```bash
# Match all materials and update database
python3 scripts/match_materials_to_mp.py
```

Or process in batches:

```bash
# Process first 100 materials
python3 scripts/match_materials_to_mp.py --limit 100
```

## Command Options

- `--dry-run` - Show matches without saving to database
- `--limit N` - Only process first N materials (useful for testing)
- `--force` - Re-match materials that already have MP IDs
- `--output FILE` - Save results to a specific JSON file (default: mp_matching_results.json)

## How Matching Works

The script uses multiple strategies to match materials:

### 1. Manual Mappings
Common commercial materials and alloys that don't have simple formulas:
- Aluminum alloys (2024-T3, 6061-T6, 7075-T6) → Pure Al
- Stainless steels (304, 316, 17-4 PH) → Pure Fe (approximation)
- Titanium alloys (Ti-6Al-4V) → Pure Ti
- Common ceramics (Alumina, SiC, ZrO2)

### 2. Formula Extraction
The script attempts to extract chemical formulas from material names:
- "Silicon Carbide" → SiC
- "Alumina 99.5%" → Al2O3
- "Al 6061-T6" → Al
- "Titanium Nitride" → TiN

### 3. Property Matching
When multiple MP entries match a formula, the script compares:
- Density (within 10% is high confidence, 30% is medium)
- Band gap (for semiconductors/insulators)
- Stability (prefers thermodynamically stable phases)
- Experimental data (prefers materials with real measurements)

## Understanding Results

After running, check `mp_matching_results.json`:

```json
{
  "material_name": "Silicon Carbide",
  "formula": "SiC",
  "mp_id": "mp-8062",
  "confidence": 0.9,
  "match_method": "formula_exact",
  "notes": "Only one match found"
}
```

### Match Methods
- `manual` - From predefined mapping (high confidence)
- `formula_exact` - Single MP entry for formula (high confidence)
- `formula_search` - Multiple entries, best match selected (medium confidence)
- `no_match` - No suitable MP entry found

### Confidence Scores
- `0.9-1.0` - High confidence (use directly)
- `0.7-0.9` - Medium confidence (verify properties)
- `0.5-0.7` - Low confidence (manual review recommended)
- `<0.5` - Very uncertain (manual matching needed)

## What Gets Updated

The script adds MP information to your materials in two ways:

### 1. In the `notes` field:
```
"notes": "High hardness ceramic | Materials Project ID: mp-8062"
```

### 2. In the `provenance` field:
```json
"provenance": {
  "mp_id": "mp-8062",
  "mp_formula": "SiC",
  "mp_match_confidence": 0.9,
  "mp_match_method": "formula_exact"
}
```

## Troubleshooting

### Rate Limiting
If you get rate limit errors, the script automatically waits 0.25s between requests. For large databases, you can:
```bash
# Process in smaller batches
python3 scripts/match_materials_to_mp.py --limit 100
# Wait a few minutes, then process next batch
python3 scripts/match_materials_to_mp.py --limit 200 --force
```

### No Matches
Common reasons materials don't match:
- **Alloys**: Commercial alloys (steel grades, aluminum alloys) don't have simple formulas. The script maps these to pure elements.
- **Polymers**: Organic polymers (PEEK, PTFE, etc.) are not in MP database.
- **Composites**: Multi-phase materials (carbon fiber epoxy) can't be matched.

### Improving Matches
To add custom mappings, edit the `manual_mappings` dictionary in the script:

```python
self.manual_mappings = {
    "Your Material Name": "mp-XXXXX",  # Add your mapping
    ...
}
```

## Example Usage

### Test First
```bash
# Set API key
export MP_API_KEY='your-key-here'

# Test with 5 materials
python3 scripts/match_materials_to_mp.py --limit 5 --dry-run

# Output shows:
# [1/5] Silicon Carbide
#   ✓ Matched: mp-8062 (confidence: 0.90)
#     Only one match found
```

### Full Run
```bash
# Match all materials
python3 scripts/match_materials_to_mp.py

# Review results
cat mp_matching_results.json | grep "mp_id" | wc -l  # Count matches
```

### Re-run with Force
```bash
# Re-match materials (e.g., after updating manual mappings)
python3 scripts/match_materials_to_mp.py --force
```

## Getting Help

The script provides detailed output showing:
- Which materials are being processed
- Formula extracted (if any)
- Match found and confidence score
- Reason for match or no-match

Check the output carefully to ensure matches make sense!
