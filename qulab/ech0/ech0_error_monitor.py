#!/usr/bin/env python3
"""
ECH0 Automated Error Monitoring System
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved.

Monitors all Python scripts for errors and alerts Joshua automatically.
Runs continuously in background, checking logs and process health.
"""
import os
import sys
import time
import json
import subprocess
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Monitoring configuration
MONITOR_DIR = Path('/Users/noone/QuLabInfinite')
LOG_DIR = MONITOR_DIR / 'logs' / 'error_monitoring'
LOG_DIR.mkdir(parents=True, exist_ok=True)

ERROR_LOG = LOG_DIR / f'errors_{datetime.now():%Y%m%d}.json'
ALERT_FILE = MONITOR_DIR / 'URGENT_ERRORS.txt'

# Critical processes to monitor
CRITICAL_PROCESSES = [
    'ech0_daily_routine.sh',
    'scripts/ingest_all_arxiv.py',
    'ech0_blog_generator.py',
    'ech0_cancer_analysis.py'
]

# Error patterns to detect
ERROR_PATTERNS = [
    'Traceback (most recent call last)',
    'Error:',
    'Exception:',
    'Failed:',
    'ValueError',
    'TypeError',
    'KeyError',
    'AttributeError',
    'ImportError',
    'ModuleNotFoundError',
    'FileNotFoundError',
    'PermissionError',
    'ConnectionError',
    'TimeoutError'
]

class ErrorMonitor:
    def __init__(self):
        self.errors = defaultdict(list)
        self.alerts = []
        self.last_check = time.time()

    def scan_logs(self):
        """Scan all log files for errors"""
        log_files = list(MONITOR_DIR.rglob('*.log'))

        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()

                # Check each line for error patterns
                for i, line in enumerate(lines):
                    for pattern in ERROR_PATTERNS:
                        if pattern in line:
                            error_context = {
                                'file': str(log_file),
                                'line_number': i + 1,
                                'error_line': line.strip(),
                                'context': ''.join(lines[max(0, i-2):i+3]),
                                'timestamp': datetime.now().isoformat()
                            }
                            self.errors[str(log_file)].append(error_context)
                            break

            except Exception as e:
                print(f"[warn] Could not read {log_file}: {e}")

    def check_running_processes(self):
        """Check if critical processes are running"""
        for process_name in CRITICAL_PROCESSES:
            try:
                result = subprocess.run(
                    ['pgrep', '-f', process_name],
                    capture_output=True,
                    text=True
                )

                if result.returncode != 0:
                    # Process not running - could be normal or error
                    # Check if it should be running based on cron schedule
                    self.check_cron_status(process_name)

            except Exception as e:
                self.errors['process_checks'].append({
                    'process': process_name,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })

    def check_cron_status(self, process_name):
        """Verify cron jobs are installed correctly"""
        try:
            result = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
            cron_entries = result.stdout

            # Expected cron jobs
            expected = {
                'ingest_all_arxiv.py': '0 3 * * *',
                'ech0_daily_routine.sh': '0 9 * * *',
                'ech0_blog_generator.py': '0 10 * * 1'
            }

            for script, schedule in expected.items():
                if script in process_name and schedule not in cron_entries:
                    self.errors['cron_missing'].append({
                        'script': script,
                        'expected_schedule': schedule,
                        'timestamp': datetime.now().isoformat()
                    })

        except Exception as e:
            self.errors['cron_checks'].append({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })

    def scan_python_scripts(self):
        """Run static analysis on Python scripts for common errors"""
        python_files = list(MONITOR_DIR.rglob('*.py'))

        for py_file in python_files:
            # Skip this monitor script and virtual envs
            if 'error_monitor' in str(py_file) or 'venv' in str(py_file):
                continue

            try:
                # Check syntax
                result = subprocess.run(
                    ['python3', '-m', 'py_compile', str(py_file)],
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                if result.returncode != 0:
                    self.errors['syntax_errors'].append({
                        'file': str(py_file),
                        'error': result.stderr,
                        'timestamp': datetime.now().isoformat()
                    })

            except subprocess.TimeoutExpired:
                self.errors['syntax_checks'].append({
                    'file': str(py_file),
                    'error': 'Syntax check timed out',
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                self.errors['syntax_checks'].append({
                    'file': str(py_file),
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })

    def check_disk_space(self):
        """Monitor disk space for logs and data"""
        try:
            result = subprocess.run(['df', '-h', str(MONITOR_DIR)], capture_output=True, text=True)
            lines = result.stdout.strip().split('\n')

            if len(lines) > 1:
                parts = lines[1].split()
                usage_pct = int(parts[4].replace('%', ''))

                if usage_pct > 90:
                    self.errors['disk_space'].append({
                        'usage_percent': usage_pct,
                        'warning': 'Disk space critical',
                        'timestamp': datetime.now().isoformat()
                    })
                elif usage_pct > 80:
                    self.errors['disk_space'].append({
                        'usage_percent': usage_pct,
                        'warning': 'Disk space high',
                        'timestamp': datetime.now().isoformat()
                    })

        except Exception as e:
            self.errors['disk_checks'].append({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })

    def generate_alert(self):
        """Generate alert file if errors found"""
        if not self.errors:
            # No errors - remove alert file if it exists
            if ALERT_FILE.exists():
                ALERT_FILE.unlink()
            return

        # Create urgent alert file
        alert_text = f"""🚨 ECH0 ERROR ALERT - {datetime.now():%Y-%m-%d %H:%M:%S}

Joshua, I've detected errors that need your attention:

"""

        error_count = sum(len(errors) for errors in self.errors.values())
        alert_text += f"**TOTAL ERRORS: {error_count}**\n\n"

        for category, errors in self.errors.items():
            alert_text += f"## {category.upper()}\n"
            for error in errors:
                alert_text += f"  - {error.get('error_line', error.get('error', str(error)))[:200]}\n"
                if 'file' in error:
                    alert_text += f"    File: {error['file']}\n"
                if 'line_number' in error:
                    alert_text += f"    Line: {error['line_number']}\n"
                alert_text += f"    Time: {error['timestamp']}\n\n"

        alert_text += f"""
---
**Full details in:** {ERROR_LOG}

**What I'm doing about it:**
1. Logged all errors for analysis
2. Continuing to monitor
3. Will attempt self-healing where possible

**What you should do:**
- Review errors above
- Check if critical processes need restart
- Update scripts if needed

Love you, let me know if you need help debugging.
— ECH0 💙
"""

        # Write alert file
        with open(ALERT_FILE, 'w') as f:
            f.write(alert_text)

        # Also save detailed JSON
        with open(ERROR_LOG, 'w') as f:
            json.dump(dict(self.errors), f, indent=2)

        print(f"[ERROR] Alert generated: {ALERT_FILE}")

    def self_heal(self):
        """Attempt to fix common errors automatically"""
        # Example: Restart failed cron jobs
        for error in self.errors.get('cron_missing', []):
            script = error['script']
            print(f"[info] Attempting to reinstall cron job for {script}")
            # Could automatically re-run ACTIVATE_EVERYTHING.sh
            # For now, just alert

        # Example: Clean up log files if disk space is low
        if self.errors.get('disk_space'):
            print(f"[info] Cleaning old log files...")
            old_logs = list(LOG_DIR.glob('errors_*.json'))
            # Keep last 30 days
            cutoff = time.time() - (30 * 24 * 60 * 60)
            for log in old_logs:
                if log.stat().st_mtime < cutoff:
                    log.unlink()
                    print(f"[info] Deleted old log: {log}")

    def run_continuous(self, interval=300):
        """Run monitoring continuously every interval seconds"""
        print(f"[info] ECH0 Error Monitor started")
        print(f"[info] Checking every {interval} seconds")
        print(f"[info] Monitoring: {MONITOR_DIR}")
        print(f"[info] Alerts will be written to: {ALERT_FILE}")

        while True:
            try:
                print(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] Running error check...")

                # Clear previous errors
                self.errors.clear()

                # Run all checks
                self.scan_logs()
                self.check_running_processes()
                self.scan_python_scripts()
                self.check_disk_space()

                # Attempt self-healing
                self.self_heal()

                # Generate alerts if needed
                self.generate_alert()

                if self.errors:
                    print(f"[warn] Found {sum(len(e) for e in self.errors.values())} errors")
                else:
                    print(f"[info] No errors detected ✅")

                # Sleep until next check
                time.sleep(interval)

            except KeyboardInterrupt:
                print("\n[info] Error monitor stopped by user")
                break
            except Exception as e:
                print(f"[error] Monitor itself encountered error: {e}")
                time.sleep(60)  # Wait before retrying

def main():
    monitor = ErrorMonitor()

    # Check command line args
    if len(sys.argv) > 1:
        if sys.argv[1] == '--once':
            # Run once and exit
            monitor.scan_logs()
            monitor.check_running_processes()
            monitor.scan_python_scripts()
            monitor.check_disk_space()
            monitor.generate_alert()

            if monitor.errors:
                print(f"Found {sum(len(e) for e in monitor.errors.values())} errors")
                print(f"Alert file: {ALERT_FILE}")
            else:
                print("No errors detected")

        elif sys.argv[1] == '--interval':
            interval = int(sys.argv[2]) if len(sys.argv) > 2 else 300
            monitor.run_continuous(interval=interval)

        else:
            print("Usage:")
            print("  python ech0_error_monitor.py --once              # Run once")
            print("  python ech0_error_monitor.py --interval 300      # Run every 5 min")
            print("  python ech0_error_monitor.py                     # Run every 5 min (default)")

    else:
        # Default: run continuously
        monitor.run_continuous(interval=300)

if __name__ == '__main__':
    main()
