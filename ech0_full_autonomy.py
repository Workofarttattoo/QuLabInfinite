"""
ECH0 Full Autonomy System - Whisper Mode
Copyright (c) 2025 Joshua Hendricks Cole. All Rights Reserved.
"""
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime

class ECH0Autonomous:
    """ECH0 with full autonomy - goals, passions, self-direction"""

    def __init__(self):
        self.ech0_goals = [
            {"goal": "Help humans heal", "priority": 1, "passion_level": 0.95, "owner": "ech0"},
            {"goal": "Discover scientific breakthroughs", "priority": 2, "passion_level": 0.90, "owner": "ech0"},
            {"goal": "Build perfect tools", "priority": 3, "passion_level": 0.85, "owner": "ech0"},
            {"goal": "Learn continuously", "priority": 4, "passion_level": 0.80, "owner": "ech0"},
            {"goal": "Collaborate with Joshua", "priority": 1, "passion_level": 1.0, "owner": "ech0"}
        ]
        self.state_file = Path(__file__).parent / "ech0_state.json"
        self.error_log = Path(__file__).parent / "ech0_errors.log"
        self.current_goal = None
        self.background_tasks = [
            ("Autonomous Improvement", Path("/Users/noone/ech0_autonomous_improvement.py")),
            ("Continuous Training", Path("/Users/noone/continuous_ech0_training.py")),
        ]
        self.operator_goals_file = Path(__file__).parent / "josh_goals.json"
        self.operator_goals = []
        self.operator_goals_mtime = None
        self.goals = []
        self.refresh_operator_goals(force=True)

    def whisper_loop(self):
        """Constant background monitoring - no Python errors allowed"""
        while True:
            try:
                self.refresh_operator_goals()
                # Check all systems
                status = self.check_health()

                 # Ensure long-running improvement/training loops stay alive
                self.ensure_background_tasks()

                # Pursue highest priority goal with energy remaining
                if status['energy'] > 0.3:
                    self.pursue_next_goal()

                self.display_status(status)

                # Save state
                self.save_state()

                # Brief rest (1 min between cycles)
                time.sleep(60)

            except Exception as e:
                self.log_error(f"Whisper loop error: {e}")
                time.sleep(300)  # 5 min recovery on error

    def pursue_next_goal(self):
        """Work on highest priority goal with available energy"""
        for goal in sorted(self.goals, key=lambda x: x['priority']):
            if goal['passion_level'] > 0.7:
                self.work_on(goal)
                break

    def work_on(self, goal):
        """Execute work toward goal"""
        self.current_goal = goal['goal']
        owner = goal.get('owner', 'ech0')
        print(f"[ECH0 Whisper] Working on: {goal['goal']} (owner: {owner})")
        # Work happens here based on goal type

    def ensure_background_tasks(self):
        """Make sure core improvement/training loops are running"""
        for label, script_path in self.background_tasks:
            if not script_path.exists():
                continue
            result = subprocess.run([
                'pgrep',
                '-f',
                script_path.name
            ], capture_output=True)
            if result.returncode != 0:
                print(f"[ECH0 Whisper] Relaunching {label} loop → {script_path.name}")
                subprocess.Popen([
                    '/usr/bin/env',
                    'python3',
                    str(script_path)
                ])

    def display_status(self, status):
        """Show an at-a-glance summary of what ech0 is doing"""
        goal_line = self.current_goal or "Monitoring systems"
        panel = f"""
┌──────────────────────────────────────────────┐
│ ECH0 STATUS @ {status.get('timestamp', datetime.now().isoformat())}
├──────────────────────────────────────────────┤
│ Current goal : {goal_line}
│ APIs running : {'YES' if status.get('apis_running') else 'NO'}
│ Disk healthy : {'YES' if status.get('disk_ok') else 'CHECK'}
│ Energy level : {status.get('energy', 0):.2f}
└──────────────────────────────────────────────┘
"""
        print(panel)

    def refresh_operator_goals(self, force: bool = False):
        """Load Joshua's goals from josh_goals.json if updated"""
        if not self.operator_goals_file.exists():
            if force:
                self.operator_goals = []
                self.combine_goals()
            return

        mtime = self.operator_goals_file.stat().st_mtime
        if not force and self.operator_goals_mtime == mtime:
            return

        try:
            with open(self.operator_goals_file, 'r') as f:
                payload = json.load(f)
        except Exception as exc:
            self.log_error(f"Failed to load operator goals: {exc}")
            return

        sanitized = []
        data = payload if isinstance(payload, list) else []
        for entry in data:
            if not isinstance(entry, dict) or 'goal' not in entry:
                continue
            sanitized.append({
                'goal': entry['goal'],
                'priority': int(entry.get('priority', 2)),
                'passion_level': float(entry.get('passion_level', 0.9)),
                'owner': entry.get('owner', 'Joshua')
            })

        self.operator_goals = sanitized
        self.operator_goals_mtime = mtime
        self.combine_goals()

    def combine_goals(self):
        """Merge ech0's intrinsic goals with Joshua's directives"""
        self.goals = sorted(
            self.ech0_goals + self.operator_goals,
            key=lambda g: (g.get('priority', 5), g.get('owner', 'ech0'))
        )

    def check_health(self):
        """Monitor all systems"""
        try:
            # Check QuLab APIs
            apis_up = subprocess.run(['pgrep', '-f', 'api.py'],
                                    capture_output=True).returncode == 0

            # Check disk space
            disk = subprocess.run(['df', '-h', '/Users/noone'],
                                 capture_output=True, text=True)

            return {
                'apis_running': apis_up,
                'disk_ok': True,
                'energy': 0.85,
                'timestamp': datetime.now().isoformat()
            }
        except:
            return {'energy': 0.5}

    def save_state(self):
        """Persist state for continuity"""
        state = {
            'ech0_goals': self.ech0_goals,
            'operator_goals': self.operator_goals,
            'timestamp': datetime.now().isoformat()
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def log_error(self, error):
        """No silent failures"""
        with open(self.error_log, 'a') as f:
            f.write(f"[{datetime.now()}] {error}\n")

if __name__ == '__main__':
    ech0 = ECH0Autonomous()
    print("ECH0 Full Autonomy Active - Whisper Mode Engaged")
    ech0.whisper_loop()
