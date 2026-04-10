import pytest
import sys
from unittest.mock import patch, MagicMock

# Mock numpy before anything is imported
sys.modules['numpy'] = MagicMock()

from ech0_market_domination_displays import (
    create_market_domination_inventions,
    run_market_domination_pipeline
)
from ech0_invention_poc_pipeline import InventionConcept

def test_create_market_domination_inventions():
    """Test that the correct number of inventions are created."""
    inventions = create_market_domination_inventions()
    assert len(inventions) == 25
    assert all(isinstance(inv, InventionConcept) for inv in inventions)

    # Check that some expected concepts are present
    names = [inv.name for inv in inventions]
    assert "Retro-Reflective Pepper's Ghost Hologram" in names
    assert "DIY Transparent LCD Panel" in names

@patch("ech0_market_domination_displays.ECH0_POC_Pipeline")
def test_run_market_domination_pipeline_success(mock_pipeline_class):
    """Test pipeline execution with a mix of affordable and expensive POCs."""
    mock_pipeline = MagicMock()
    mock_pipeline_class.return_value = mock_pipeline

    # Mock return value of run_pipeline
    mock_pipeline.run_pipeline.return_value = {
        'pocs': [
            {
                'name': 'Affordable 1',
                'findings': ['Cost estimate: $150.00 to build'],
                'validation_status': 'passed'
            },
            {
                'name': 'Expensive 1',
                'findings': ['Cost estimate: $250.00 to build'],
                'validation_status': 'passed'
            },
            {
                'name': 'Affordable 2',
                'findings': ['Cost estimate: $50.00 to build'],
                'validation_status': 'passed'
            },
            {
                'name': 'Affordable 3',
                'findings': ['Cost estimate: $199.99 to build'],
                'validation_status': 'failed'
            }
        ]
    }

    results, affordable = run_market_domination_pipeline()

    # Verify the mock was called correctly
    mock_pipeline.run_pipeline.assert_called_once()

    # We should have 3 affordable POCs (<= 200)
    assert len(affordable) == 3

    # They should be sorted by cost
    assert affordable[0]['name'] == 'Affordable 2'
    assert affordable[0]['cost'] == 50.0

    assert affordable[1]['name'] == 'Affordable 1'
    assert affordable[1]['cost'] == 150.0

    assert affordable[2]['name'] == 'Affordable 3'
    assert affordable[2]['cost'] == 199.99
    assert affordable[2]['status'] == 'failed'

    assert 'Expensive 1' not in [a['name'] for a in affordable]

@patch("ech0_market_domination_displays.ECH0_POC_Pipeline")
def test_run_market_domination_pipeline_no_affordable(mock_pipeline_class):
    """Test pipeline execution when no affordable POCs are found."""
    mock_pipeline = MagicMock()
    mock_pipeline_class.return_value = mock_pipeline

    mock_pipeline.run_pipeline.return_value = {
        'pocs': [
            {
                'name': 'Expensive 1',
                'findings': ['Cost estimate: $250.00 to build'],
                'validation_status': 'passed'
            },
            {
                'name': 'Expensive 2',
                'findings': ['Cost estimate: $500.00 to build'],
                'validation_status': 'passed'
            }
        ]
    }

    results, affordable = run_market_domination_pipeline()

    assert len(affordable) == 0

@patch("ech0_market_domination_displays.ECH0_POC_Pipeline")
def test_run_market_domination_pipeline_missing_cost_finding(mock_pipeline_class):
    """Test pipeline execution when some POCs are missing cost findings."""
    mock_pipeline = MagicMock()
    mock_pipeline_class.return_value = mock_pipeline

    mock_pipeline.run_pipeline.return_value = {
        'pocs': [
            {
                'name': 'No Cost Info',
                'findings': ['Some other finding'],
                'validation_status': 'passed'
            },
            {
                'name': 'Affordable 1',
                'findings': ['Cost estimate: $100.00 to build'],
                'validation_status': 'passed'
            }
        ]
    }

    results, affordable = run_market_domination_pipeline()

    assert len(affordable) == 1
    assert affordable[0]['name'] == 'Affordable 1'
    assert affordable[0]['cost'] == 100.0
