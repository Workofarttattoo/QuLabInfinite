#!/bin/bash
# Installation script for 2026 cutting-edge integrations
# QuLabInfinite Enhancement Plan - Phase 1

set -e

echo "================================================================================"
echo "QuLabInfinite 2026 Tools Installation"
echo "================================================================================"
echo ""

# Check Python version
python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✅ Python version: $python_version"

if [ "$(printf '%s\n' "3.8" "$python_version" | sort -V | head -n1)" != "3.8" ]; then
    echo "❌ Python 3.8+ required"
    exit 1
fi

echo ""
echo "================================================================================"
echo "Phase 1: Data Processing (10-100x Faster)"
echo "================================================================================"
echo ""

# Install Polars
echo "📦 Installing Polars (Rust-based DataFrame)..."
pip install --user polars
echo "✅ Polars installed"
echo ""

# Install DuckDB
echo "📦 Installing DuckDB (SQL interface)..."
pip install --user duckdb
echo "✅ DuckDB installed"
echo ""

# Install Pandas (for DuckDB integration)
echo "📦 Installing Pandas (for compatibility)..."
pip install --user pandas
echo "✅ Pandas installed"
echo ""

echo "================================================================================"
echo "Phase 2: Machine Learning (Optional - Large download)"
echo "================================================================================"
echo ""

read -p "Install MatGL for ML property prediction? (requires PyTorch ~2GB) [y/N]: " install_matgl

if [[ $install_matgl =~ ^[Yy]$ ]]; then
    echo "📦 Installing MatGL + PyTorch (this may take a while)..."
    pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install --user matgl
    echo "✅ MatGL installed"
else
    echo "⏭️  Skipping MatGL (can install later with: pip install matgl torch)"
fi

echo ""
echo "================================================================================"
echo "Installation Summary"
echo "================================================================================"
echo ""

# Test installations
echo "Testing installations..."
echo ""

python3 -c "import polars; print('✅ Polars:', polars.__version__)"
python3 -c "import duckdb; print('✅ DuckDB:', duckdb.__version__)"
python3 -c "import pandas; print('✅ Pandas:', pandas.__version__)"

if [[ $install_matgl =~ ^[Yy]$ ]]; then
    python3 -c "import torch; print('✅ PyTorch:', torch.__version__)" 2>/dev/null || echo "⚠️  PyTorch installation may need verification"
    python3 -c "import matgl; print('✅ MatGL:', matgl.__version__)" 2>/dev/null || echo "⚠️  MatGL installation may need verification"
fi

echo ""
echo "================================================================================"
echo "Next Steps"
echo "================================================================================"
echo ""
echo "1. Test Polars fast screening:"
echo "   python3 materials_lab/materials_screening_fast.py"
echo ""
echo "2. Test DuckDB SQL interface:"
echo "   python3 materials_lab/materials_sql.py"
echo ""
if [[ $install_matgl =~ ^[Yy]$ ]]; then
    echo "3. Test ML property prediction:"
    echo "   python3 materials_lab/ml_property_predictor.py"
    echo ""
fi
echo "4. For full database (1.4M materials):"
echo "   Copy extended_materials_db.json (14GB) to: data/"
echo ""
echo "5. Read documentation:"
echo "   - ENHANCEMENT_PLAN_2026.md (roadmap)"
echo "   - QUICK_START_INTEGRATIONS.md (examples)"
echo "   - MATERIALS_DATABASE.md (database info)"
echo ""
echo "================================================================================"
echo "🎉 Installation Complete!"
echo "================================================================================"
