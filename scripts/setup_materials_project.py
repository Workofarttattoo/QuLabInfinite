#!/usr/bin/env python3
"""
Materials Project Setup Script

Interactive setup for Materials Project API integration.
This script helps you:
1. Sign up for Materials Project API (if needed)
2. Configure your API key
3. Test the connection
4. Download the initial dataset
"""

import os
import sys
from pathlib import Path
import subprocess
import webbrowser

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text):
    """Print formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 80}")
    print(f"{text}")
    print(f"{'=' * 80}{Colors.END}\n")


def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_info(text):
    """Print info message"""
    print(f"{Colors.CYAN}ℹ {text}{Colors.END}")


def print_warning(text):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


def check_pymatgen():
    """Check if pymatgen is installed"""
    try:
        import pymatgen
        return True
    except ImportError:
        return False


def install_pymatgen():
    """Install pymatgen"""
    print_info("Installing pymatgen (Materials Project Python library)...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pymatgen"])
        print_success("pymatgen installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print_error("Failed to install pymatgen")
        return False


def open_signup_page():
    """Open Materials Project signup page"""
    url = "https://materialsproject.org/api"
    print_info(f"Opening {url} in your browser...")
    try:
        webbrowser.open(url)
        return True
    except Exception as e:
        print_warning(f"Could not open browser automatically: {e}")
        print_info(f"Please visit manually: {url}")
        return False


def get_api_key():
    """Prompt user for API key"""
    print_info("Enter your Materials Project API key:")
    print_info("(You can find it at: https://materialsproject.org/api)")
    api_key = input(f"{Colors.CYAN}MP_API_KEY: {Colors.END}").strip()
    return api_key


def save_api_key(api_key: str):
    """Save API key to .env file"""
    env_path = Path(__file__).parent.parent / ".env"
    env_example_path = Path(__file__).parent.parent / ".env.example"

    # If .env doesn't exist, copy from .env.example
    if not env_path.exists() and env_example_path.exists():
        with open(env_example_path, 'r') as f:
            content = f.read()
        with open(env_path, 'w') as f:
            f.write(content)
        print_info("Created .env file from .env.example")

    # Update or add MP_API_KEY
    if env_path.exists():
        with open(env_path, 'r') as f:
            lines = f.readlines()

        # Find and replace MP_API_KEY line
        found = False
        for i, line in enumerate(lines):
            if line.strip().startswith("MP_API_KEY"):
                lines[i] = f"MP_API_KEY={api_key}\n"
                found = True
                break

        # If not found, add it
        if not found:
            lines.append(f"\nMP_API_KEY={api_key}\n")

        with open(env_path, 'w') as f:
            f.writelines(lines)

        print_success(f"API key saved to {env_path}")
    else:
        # Create new .env file
        with open(env_path, 'w') as f:
            f.write(f"# QuLabInfinite Environment Configuration\n\n")
            f.write(f"MP_API_KEY={api_key}\n")

        print_success(f"Created {env_path} with API key")

    # Add to current environment
    os.environ["MP_API_KEY"] = api_key


def test_connection(api_key: str):
    """Test Materials Project API connection"""
    print_info("Testing connection to Materials Project...")

    # Set API key for this session
    os.environ["MP_API_KEY"] = api_key

    # Add project root to path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    try:
        from materials_lab.materials_project_client import MaterialsProjectClient

        client = MaterialsProjectClient(api_key=api_key)

        # Try to fetch Silicon (mp-149)
        material = client.get_material("mp-149")

        if material:
            print_success("Connection successful!")
            print_info(f"Test material: {material.formula} (mp-149)")
            print_info(f"  Density: {material.density:.2f} g/cm³")
            print_info(f"  Band gap: {material.band_gap:.2f} eV")
            return True
        else:
            print_error("Connection failed: Could not fetch test material")
            return False

    except Exception as e:
        print_error(f"Connection test failed: {e}")
        return False


def download_dataset():
    """Ask user if they want to download the common materials dataset"""
    print_info("\nWould you like to download the 100 common materials dataset now?")
    print_info("This will take about 5-10 minutes and use ~5 MB of disk space.")

    response = input(f"{Colors.CYAN}Download now? (y/n): {Colors.END}").strip().lower()

    if response == 'y':
        print_info("Starting download...")
        print_info("This may take a few minutes...\n")

        script_path = Path(__file__).parent / "download_common_materials.py"

        try:
            subprocess.check_call([sys.executable, str(script_path)])
            print_success("\nDataset download complete!")
            return True
        except subprocess.CalledProcessError:
            print_error("\nDataset download failed")
            return False
    else:
        print_info("Skipping dataset download")
        print_info(f"You can download it later by running:")
        print_info(f"  python scripts/download_common_materials.py")
        return False


def main():
    """Main setup workflow"""
    print_header("MATERIALS PROJECT SETUP")
    print("This script will help you set up Materials Project integration for QuLabInfinite.\n")

    # Step 1: Check pymatgen
    print_header("STEP 1: Check Dependencies")
    if check_pymatgen():
        print_success("pymatgen is already installed")
    else:
        print_warning("pymatgen is not installed")
        response = input(f"{Colors.CYAN}Install pymatgen now? (y/n): {Colors.END}").strip().lower()
        if response == 'y':
            if not install_pymatgen():
                print_error("Setup cannot continue without pymatgen")
                sys.exit(1)
        else:
            print_error("Setup cannot continue without pymatgen")
            sys.exit(1)

    # Step 2: Check for existing API key
    print_header("STEP 2: Materials Project API Key")

    existing_key = os.environ.get("MP_API_KEY")
    if existing_key and existing_key != "your_materials_project_api_key_here":
        print_success("Found existing API key in environment")
        response = input(f"{Colors.CYAN}Test existing key? (y/n): {Colors.END}").strip().lower()
        if response == 'y':
            if test_connection(existing_key):
                api_key = existing_key
            else:
                print_warning("Existing key doesn't work. Let's set up a new one.")
                api_key = None
        else:
            api_key = existing_key
    else:
        api_key = None

    # Step 3: Get API key
    if not api_key:
        print_info("\n📝 To get a Materials Project API key:")
        print_info("   1. Visit https://materialsproject.org")
        print_info("   2. Sign up or log in (it's free!)")
        print_info("   3. Go to https://materialsproject.org/api")
        print_info("   4. Copy your API key\n")

        response = input(f"{Colors.CYAN}Open signup page in browser? (y/n): {Colors.END}").strip().lower()
        if response == 'y':
            open_signup_page()

        print()
        api_key = get_api_key()

        if not api_key:
            print_error("No API key provided. Setup cancelled.")
            sys.exit(1)

        # Save API key
        save_api_key(api_key)

    # Step 4: Test connection
    print_header("STEP 3: Test Connection")
    if not test_connection(api_key):
        print_error("\nSetup failed: Could not connect to Materials Project")
        print_info("Please check your API key and try again")
        sys.exit(1)

    # Step 5: Download dataset
    print_header("STEP 4: Download Dataset (Optional)")
    download_dataset()

    # Final message
    print_header("SETUP COMPLETE!")
    print_success("Materials Project integration is ready to use!")

    print(f"\n{Colors.BOLD}Next Steps:{Colors.END}")
    print("1. Run validation tests:")
    print("   python -m pytest materials_lab/tests/test_materials_project.py")
    print("\n2. Try the Materials Project client:")
    print("   python materials_lab/materials_project_client.py")
    print("\n3. Validate your aerogel simulations:")
    print("   python materials_lab/materials_validator.py")
    print("\n4. Download more materials:")
    print("   python scripts/download_common_materials.py")

    print(f"\n{Colors.CYAN}Happy simulating! 🧪{Colors.END}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Setup cancelled by user{Colors.END}")
        sys.exit(0)
    except Exception as e:
        print_error(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
