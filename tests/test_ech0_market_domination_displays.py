import pytest
import sys
from unittest.mock import patch, MagicMock

# Mock numpy to allow import
sys.modules['numpy'] = MagicMock()

from ech0_market_domination_displays import run_market_domination_pipeline, create_market_domination_inventions

@patch('ech0_market_domination_displays.ECH0_POC_Pipeline')
@patch('ech0_market_domination_displays.create_market_domination_inventions')
def test_run_market_domination_pipeline(mock_create_inventions, mock_pipeline_class):
    # Setup mock inventions
    mock_invention = MagicMock()
    mock_invention.name = "Test Invention"
    mock_create_inventions.return_value = [mock_invention]

    # Setup mock pipeline
    mock_pipeline_instance = MagicMock()
    mock_pipeline_class.return_value = mock_pipeline_instance

    # Mock run_pipeline output
    mock_results = {
        'pocs': [
            {
                'name': 'Affordable Passed',
                'validation_status': 'passed',
                'findings': ['Cost estimate $150.00 within budget']
            },
            {
                'name': 'Expensive Passed',
                'validation_status': 'passed',
                'findings': ['Cost estimate $250.00 exceeds budget']
            },
            {
                'name': 'Affordable Failed',
                'validation_status': 'needs_work',
                'findings': ['Cost estimate $100.00 within budget']
            },
            {
                'name': 'No Cost Finding',
                'validation_status': 'passed',
                'findings': ['Other finding']
            }
        ]
    }
    mock_pipeline_instance.run_pipeline.return_value = mock_results

    # Run the function
    results, affordable = run_market_domination_pipeline()

    # Assertions
    # 1. Pipeline should be instantiated
    mock_pipeline_class.assert_called_once()

    # 2. Inventions should be passed to pipeline with requirements
    mock_pipeline_instance.run_pipeline.assert_called_once()
    args, kwargs = mock_pipeline_instance.run_pipeline.call_args
    assert args[0] == [mock_invention]
    assert 'application' in args[1]
    assert args[1]['budget'] == 200.0

    # 3. Check affordable extraction logic (should sort by cost)
    assert len(affordable) == 2

    # Affordable Failed is $100
    assert affordable[0]['name'] == 'Affordable Failed'
    assert affordable[0]['cost'] == 100.0
    assert affordable[0]['status'] == 'needs_work'

    # Affordable Passed is $150
    assert affordable[1]['name'] == 'Affordable Passed'
    assert affordable[1]['cost'] == 150.0
    assert affordable[1]['status'] == 'passed'

@patch('ech0_market_domination_displays.ECH0_POC_Pipeline')
@patch('ech0_market_domination_displays.create_market_domination_inventions')
def test_run_market_domination_pipeline_empty(mock_create_inventions, mock_pipeline_class):
    # Setup mock inventions
    mock_create_inventions.return_value = []

    # Setup mock pipeline
    mock_pipeline_instance = MagicMock()
    mock_pipeline_class.return_value = mock_pipeline_instance

    # Mock run_pipeline output
    mock_results = {'pocs': []}
    mock_pipeline_instance.run_pipeline.return_value = mock_results

    # Run the function
    results, affordable = run_market_domination_pipeline()

    # Assertions
    assert len(affordable) == 0
    assert results == mock_results
