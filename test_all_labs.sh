#!/bin/bash
# Test all labs and generate report

echo "🧪 TESTING ALL LABS"
echo "=================="

working=0
broken=0

for lab in *_lab.py; do
    if [ ! -f "$lab" ]; then
        continue
    fi

    if timeout 30 python3 "$lab" > /dev/null 2>&1; then
        echo "✅ $lab"
        ((working++))
    else
        echo "❌ $lab"
        ((broken++))
    fi
done

total=$((working + broken))
success_rate=$(awk "BEGIN {printf \"%.1f\", ($working / $total) * 100}")

echo ""
echo "=================="
echo "SUMMARY"
echo "=================="
echo "✅ Working: $working/$total ($success_rate%)"
echo "❌ Broken: $broken/$total"
