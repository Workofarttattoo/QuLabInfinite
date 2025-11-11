"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

QuLabInfinite Master API - Usage Examples
==========================================

This file demonstrates comprehensive usage of the QuLabInfinite Master API,
showing how to integrate all 80+ scientific laboratories in your applications.
"""

from qulab_master_api import QuLabMasterAPI, LabDomain
import json


def example_1_basic_initialization():
    """Example 1: Basic initialization and statistics"""
    print("\n" + "="*80)
    print("Example 1: Basic Initialization")
    print("="*80 + "\n")

    # Initialize the API
    api = QuLabMasterAPI(auto_load=True, verbose=False)

    # Get statistics
    stats = api.get_statistics()

    print(f"Total Labs: {stats['total_labs']}")
    print(f"Available Labs: {stats['available_labs']}")
    print(f"Success Rate: {stats['success_rate']}")
    print(f"\nLabs by Domain:")
    for domain, counts in sorted(stats['by_domain'].items()):
        print(f"  {domain}: {counts['available']}/{counts['total']}")


def example_2_listing_labs():
    """Example 2: Listing labs with filters"""
    print("\n" + "="*80)
    print("Example 2: Listing Labs")
    print("="*80 + "\n")

    api = QuLabMasterAPI(auto_load=True, verbose=False)

    # List all available physics labs
    print("Available Physics Labs:")
    print("-" * 40)
    physics_labs = api.list_labs(domain=LabDomain.PHYSICS, available_only=True)
    for lab in physics_labs:
        print(f"  • {lab['display_name']}")
        print(f"    {lab['description']}")
        print()

    # List all biology labs (including unavailable)
    print("\nAll Biology Labs:")
    print("-" * 40)
    bio_labs = api.list_labs(domain=LabDomain.BIOLOGY, available_only=False)
    for lab in bio_labs:
        status = "✓" if lab['available'] else "✗"
        print(f"  {status} {lab['display_name']}")


def example_3_searching_labs():
    """Example 3: Searching for labs"""
    print("\n" + "="*80)
    print("Example 3: Searching Labs")
    print("="*80 + "\n")

    api = QuLabMasterAPI(auto_load=True, verbose=False)

    # Search for quantum-related labs
    print("Search: 'quantum'")
    print("-" * 40)
    results = api.search_labs("quantum")
    for result in results[:5]:
        print(f"  • {result['display_name']} (Relevance: {result['relevance_score']:.1f})")
        print(f"    {result['description']}")
        print()

    # Search for cancer research labs
    print("\nSearch: 'cancer'")
    print("-" * 40)
    results = api.search_labs("cancer")
    for result in results[:5]:
        print(f"  • {result['display_name']} (Relevance: {result['relevance_score']:.1f})")


def example_4_lab_capabilities():
    """Example 4: Inspecting lab capabilities"""
    print("\n" + "="*80)
    print("Example 4: Lab Capabilities")
    print("="*80 + "\n")

    api = QuLabMasterAPI(auto_load=True, verbose=False)

    # Check materials lab capabilities
    lab_name = "materials_lab"
    capabilities = api.get_capabilities(lab_name)

    print(f"Lab: {capabilities['display_name']}")
    print(f"Description: {capabilities['description']}")
    print(f"Domain: {capabilities['domain']}")
    print(f"Available: {capabilities['available']}")

    if capabilities['available']:
        print(f"\nCapabilities ({len(capabilities['capabilities'])}):")
        for cap in capabilities['capabilities'][:10]:
            print(f"  • {cap}")
        if len(capabilities['capabilities']) > 10:
            print(f"  ... and {len(capabilities['capabilities']) - 10} more")

    print(f"\nKeywords: {', '.join(capabilities['keywords'])}")


def example_5_using_lab_instance():
    """Example 5: Getting and using a lab instance"""
    print("\n" + "="*80)
    print("Example 5: Using Lab Instance")
    print("="*80 + "\n")

    api = QuLabMasterAPI(auto_load=True, verbose=False)

    # Get thermodynamics lab
    lab = api.get_lab("thermodynamics")

    if lab:
        print(f"Successfully loaded: {lab.__class__.__name__}")
        print(f"Available methods: {[m for m in dir(lab) if not m.startswith('_')][:10]}")

        # Try to use it
        try:
            if hasattr(lab, 'calculate_entropy'):
                result = lab.calculate_entropy(temperature=298.15, volume=1.0)
                print(f"\nExample calculation - Entropy: {result}")
        except Exception as e:
            print(f"\nNote: {e}")
    else:
        print("Lab not available")


def example_6_running_demo():
    """Example 6: Running lab demonstrations"""
    print("\n" + "="*80)
    print("Example 6: Running Demos")
    print("="*80 + "\n")

    api = QuLabMasterAPI(auto_load=True, verbose=False)

    # Try to run demo for available labs
    test_labs = ["materials_lab", "nuclear_physics", "thermodynamics"]

    for lab_name in test_labs:
        print(f"\nTrying demo for: {lab_name}")
        print("-" * 40)

        result = api.run_demo(lab_name)

        if "error" in result:
            print(f"  Error: {result['error']}")
            if "suggestion" in result:
                print(f"  Suggestion: {result['suggestion']}")
        else:
            print(f"  Success: {result.get('success', False)}")
            if 'result' in result:
                print(f"  Result type: {type(result['result'])}")


def example_7_domain_analysis():
    """Example 7: Analyzing labs by domain"""
    print("\n" + "="*80)
    print("Example 7: Domain Analysis")
    print("="*80 + "\n")

    api = QuLabMasterAPI(auto_load=True, verbose=False)

    stats = api.get_statistics()

    print("Lab Distribution by Domain:")
    print("-" * 60)
    print(f"{'Domain':<30} {'Available':<12} {'Total':<12} {'Rate':<10}")
    print("-" * 60)

    for domain, counts in sorted(stats['by_domain'].items()):
        rate = (counts['available'] / counts['total'] * 100) if counts['total'] > 0 else 0
        print(f"{domain:<30} {counts['available']:<12} {counts['total']:<12} {rate:.1f}%")


def example_8_exporting_catalog():
    """Example 8: Exporting lab catalog"""
    print("\n" + "="*80)
    print("Example 8: Exporting Catalog")
    print("="*80 + "\n")

    api = QuLabMasterAPI(auto_load=True, verbose=False)

    # Export to file
    output_path = "/tmp/qulab_catalog.json"
    catalog_json = api.export_catalog(output_path)

    print(f"Catalog exported to: {output_path}")

    # Parse and show summary
    catalog = json.loads(catalog_json)
    print(f"\nCatalog Summary:")
    print(f"  Title: {catalog['metadata']['title']}")
    print(f"  Version: {catalog['metadata']['version']}")
    print(f"  Total Labs: {catalog['metadata']['total_labs']}")
    print(f"  Generated: {catalog['metadata']['generated']}")


def example_9_custom_lab_workflow():
    """Example 9: Custom workflow with multiple labs"""
    print("\n" + "="*80)
    print("Example 9: Custom Workflow")
    print("="*80 + "\n")

    api = QuLabMasterAPI(auto_load=True, verbose=False)

    # Workflow: Find all available medicine labs and get their capabilities
    medicine_labs = api.list_labs(domain=LabDomain.MEDICINE, available_only=True)

    print(f"Found {len(medicine_labs)} available medicine labs\n")

    for lab_info in medicine_labs:
        lab_name = lab_info['name']
        capabilities = api.get_capabilities(lab_name)

        print(f"Lab: {lab_info['display_name']}")
        print(f"  Capabilities: {capabilities['capabilities_count']}")
        print(f"  Keywords: {', '.join(capabilities['keywords'][:3])}")
        print()


def example_10_batch_processing():
    """Example 10: Batch processing multiple labs"""
    print("\n" + "="*80)
    print("Example 10: Batch Processing")
    print("="*80 + "\n")

    api = QuLabMasterAPI(auto_load=True, verbose=False)

    # Get all available labs
    all_labs = api.list_labs(available_only=True)

    print(f"Processing {len(all_labs)} available labs...\n")

    results = []
    for lab_info in all_labs[:5]:  # Process first 5 for demo
        lab_name = lab_info['name']
        capabilities = api.get_capabilities(lab_name)

        results.append({
            'name': lab_name,
            'display_name': lab_info['display_name'],
            'domain': lab_info['domain'],
            'capability_count': len(capabilities['capabilities'])
        })

    # Sort by capability count
    results.sort(key=lambda x: x['capability_count'], reverse=True)

    print("Labs sorted by capability count:")
    print("-" * 60)
    for r in results:
        print(f"  {r['display_name']:<40} {r['capability_count']:>3} capabilities")


def example_11_programmatic_lab_selection():
    """Example 11: Programmatic lab selection"""
    print("\n" + "="*80)
    print("Example 11: Programmatic Lab Selection")
    print("="*80 + "\n")

    api = QuLabMasterAPI(auto_load=True, verbose=False)

    # Find the best lab for a specific task
    search_term = "molecular"
    results = api.search_labs(search_term)

    if results:
        best_match = results[0]
        print(f"Best match for '{search_term}':")
        print(f"  Lab: {best_match['display_name']}")
        print(f"  Description: {best_match['description']}")
        print(f"  Domain: {best_match['domain']}")
        print(f"  Relevance: {best_match['relevance_score']:.1f}")
        print(f"  Available: {best_match['available']}")

        if best_match['available']:
            # Try to get the instance
            lab = api.get_lab(best_match['name'])
            if lab:
                print(f"\n  Successfully loaded lab instance")
                print(f"  Class: {lab.__class__.__name__}")
    else:
        print(f"No labs found matching '{search_term}'")


def example_12_error_handling():
    """Example 12: Proper error handling"""
    print("\n" + "="*80)
    print("Example 12: Error Handling")
    print("="*80 + "\n")

    api = QuLabMasterAPI(auto_load=True, verbose=False)

    # Try to get a non-existent lab
    print("Attempting to get non-existent lab:")
    lab = api.get_lab("nonexistent_lab")
    if lab is None:
        print("  ✓ Properly returned None for missing lab")

    # Try to get capabilities of unavailable lab
    print("\nGetting capabilities of unavailable lab:")
    capabilities = api.get_capabilities("quantum_mechanics")
    if not capabilities['available']:
        print(f"  ✓ Lab marked as unavailable")
        print(f"  Error: {capabilities['error']}")

    # Try to run demo on unavailable lab
    print("\nAttempting demo on unavailable lab:")
    result = api.run_demo("quantum_mechanics")
    if "error" in result:
        print(f"  ✓ Proper error handling: {result['error']}")


def run_all_examples():
    """Run all examples"""
    print("\n" + "="*80)
    print("QuLabInfinite Master API - Comprehensive Examples")
    print("="*80)

    examples = [
        example_1_basic_initialization,
        example_2_listing_labs,
        example_3_searching_labs,
        example_4_lab_capabilities,
        example_5_using_lab_instance,
        example_6_running_demo,
        example_7_domain_analysis,
        example_8_exporting_catalog,
        example_9_custom_lab_workflow,
        example_10_batch_processing,
        example_11_programmatic_lab_selection,
        example_12_error_handling,
    ]

    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\nError in {example_func.__name__}: {e}")

    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Run all examples
    run_all_examples()

    # Or run individual examples:
    # example_1_basic_initialization()
    # example_3_searching_labs()
    # example_7_domain_analysis()
