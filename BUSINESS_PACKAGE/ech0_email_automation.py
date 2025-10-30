#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

ECH0 Email Automation System
Automatically send emails based on customer triggers using Mail.app
"""

import json
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path


class ECH0_EmailAutomation:
    """
    Automated email system for QuLabInfinite business operations.

    Features:
    - Load templates from ech0_email_templates.json
    - Variable substitution
    - Trigger-based automation
    - Send via Mail.app
    - Track sent emails
    """

    def __init__(self, templates_file: str = "ech0_email_templates.json"):
        """Initialize email automation system."""
        self.templates_file = templates_file
        self.templates = self._load_templates()
        self.sent_log = []

    def _load_templates(self) -> Dict[str, Any]:
        """Load email templates from JSON file."""
        templates_path = Path(__file__).parent / self.templates_file

        if not templates_path.exists():
            raise FileNotFoundError(f"Templates file not found: {templates_path}")

        with open(templates_path, 'r') as f:
            return json.load(f)

    def fill_template(self,
                     template_name: str,
                     customer_data: Dict[str, Any]) -> tuple[str, str]:
        """
        Fill template with customer data.

        Args:
            template_name: Name of template (e.g., 'cold_outreach_materials_scientists')
            customer_data: Dict with customer variables

        Returns:
            Tuple of (subject, body) with variables filled
        """
        if template_name not in self.templates['templates']:
            raise ValueError(f"Template not found: {template_name}")

        template = self.templates['templates'][template_name]
        subject = template['subject']
        body = template['body']

        # Replace variables
        for key, value in customer_data.items():
            placeholder = f"{{{{{key}}}}}"
            subject = subject.replace(placeholder, str(value))
            body = body.replace(placeholder, str(value))

        # Replace global variables
        for key, value in self.templates['variables'].items():
            placeholder = f"{{{{{key}}}}}"
            subject = subject.replace(placeholder, str(value))
            body = body.replace(placeholder, str(value))

        return subject, body

    def send_email(self,
                   to: str,
                   subject: str,
                   body: str,
                   dry_run: bool = False) -> bool:
        """
        Send email via Mail.app.

        Args:
            to: Recipient email address
            subject: Email subject
            body: Email body
            dry_run: If True, print email instead of sending

        Returns:
            True if sent successfully
        """
        if dry_run:
            print(f"\n{'='*70}")
            print(f"DRY RUN - Email would be sent to: {to}")
            print(f"{'='*70}")
            print(f"Subject: {subject}")
            print(f"\nBody:\n{body}")
            print(f"{'='*70}\n")
            return True

        try:
            # Escape quotes for AppleScript
            subject_escaped = subject.replace('"', '\\"')
            body_escaped = body.replace('"', '\\"').replace('\n', '\\n')

            applescript = f'''
            tell application "Mail"
                set newMessage to make new outgoing message with properties {{
                    subject:"{subject_escaped}",
                    content:"{body_escaped}",
                    visible:false
                }}
                tell newMessage
                    make new to recipient with properties {{address:"{to}"}}
                    send
                end tell
            end tell
            '''

            subprocess.run(['osascript', '-e', applescript], check=True)

            # Log sent email
            self.sent_log.append({
                'to': to,
                'subject': subject,
                'timestamp': datetime.now().isoformat()
            })

            print(f"✅ Email sent to {to}: {subject}")
            return True

        except Exception as e:
            print(f"❌ Failed to send email to {to}: {e}")
            return False

    def send_template(self,
                     template_name: str,
                     customer_data: Dict[str, Any],
                     dry_run: bool = False) -> bool:
        """
        Send email from template.

        Args:
            template_name: Name of template
            customer_data: Must include 'email' key
            dry_run: If True, don't actually send

        Returns:
            True if sent successfully
        """
        if 'email' not in customer_data:
            raise ValueError("customer_data must include 'email' key")

        subject, body = self.fill_template(template_name, customer_data)
        return self.send_email(customer_data['email'], subject, body, dry_run)

    def check_triggers(self,
                      customers: List[Dict[str, Any]],
                      dry_run: bool = False) -> List[Dict[str, Any]]:
        """
        Check all customers for automation triggers and send appropriate emails.

        Args:
            customers: List of customer dicts with:
                - email, first_name, trial_created_date, api_requests_count, etc.
            dry_run: If True, don't actually send emails

        Returns:
            List of emails sent
        """
        emails_sent = []

        for customer in customers:
            # Check each automation rule
            for rule_name, rule in self.templates['automation_rules'].items():
                if self._should_trigger(rule, customer):
                    # Send email
                    success = self.send_template(
                        rule['template'],
                        customer,
                        dry_run=dry_run
                    )

                    if success:
                        emails_sent.append({
                            'customer': customer['email'],
                            'rule': rule_name,
                            'template': rule['template']
                        })

        return emails_sent

    def _should_trigger(self,
                       rule: Dict[str, Any],
                       customer: Dict[str, Any]) -> bool:
        """Check if automation rule should trigger for customer."""
        trigger = rule['trigger']
        conditions = rule.get('conditions')

        # Check trigger type
        triggered = False

        if trigger == 'trial_created':
            # Send immediately when trial is created
            # (Check if email hasn't been sent yet)
            triggered = customer.get('trial_welcome_sent') != True

        elif trigger == 'trial_day_3':
            trial_date = datetime.fromisoformat(customer.get('trial_created_date', ''))
            days_since_trial = (datetime.now() - trial_date).days
            triggered = days_since_trial == 3 and not customer.get('day3_email_sent')

        elif trigger == 'trial_day_7':
            trial_date = datetime.fromisoformat(customer.get('trial_created_date', ''))
            days_since_trial = (datetime.now() - trial_date).days
            triggered = days_since_trial == 7 and not customer.get('day7_email_sent')

        elif trigger == 'trial_day_12':
            trial_date = datetime.fromisoformat(customer.get('trial_created_date', ''))
            days_since_trial = (datetime.now() - trial_date).days
            triggered = days_since_trial == 12 and not customer.get('day12_email_sent')

        elif trigger == 'trial_expired_no_payment':
            trial_date = datetime.fromisoformat(customer.get('trial_created_date', ''))
            days_since_trial = (datetime.now() - trial_date).days
            has_paid = customer.get('payment_received', False)
            triggered = days_since_trial >= 14 and not has_paid and not customer.get('expiration_email_sent')

        elif trigger == 'payment_received':
            triggered = customer.get('payment_received') == True and not customer.get('welcome_email_sent')

        elif trigger == 'customer_anniversary_month':
            signup_date_str = customer.get('signup_date', '')
            if signup_date_str:
                signup_date = datetime.fromisoformat(signup_date_str)
                months_since = (datetime.now() - signup_date).days // 30
                last_checkin = customer.get('last_monthly_checkin_month', 0)
                triggered = months_since > last_checkin
            else:
                triggered = False

        elif trigger == 'api_limit_approached':
            tier = customer.get('tier', '')
            usage_pct = customer.get('usage_percentage', 0)
            triggered = tier == 'starter' and usage_pct >= 80 and not customer.get('upsell_email_sent')

        elif trigger == 'usage_drop':
            prev_usage = customer.get('prev_month_requests', 0)
            curr_usage = customer.get('current_month_requests', 0)
            drop_pct = ((prev_usage - curr_usage) / prev_usage * 100) if prev_usage > 0 else 0
            triggered = drop_pct >= 70 and not customer.get('churn_risk_email_sent')

        # Check conditions if triggered
        if triggered and conditions:
            for condition_key, condition_value in conditions.items():
                customer_value = customer.get(condition_key.replace('_min', '').replace('_max', ''), 0)

                if '_min' in condition_key:
                    if customer_value < condition_value:
                        return False
                elif '_max' in condition_key:
                    if customer_value > condition_value:
                        return False
                else:
                    if customer_value != condition_value:
                        return False

        return triggered

    def save_log(self, filepath: str = "email_automation_log.json"):
        """Save sent email log to file."""
        with open(filepath, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total_sent': len(self.sent_log),
                'emails': self.sent_log
            }, f, indent=2)

        print(f"✅ Email log saved to: {filepath}")


def demo_email_automation():
    """Demo: Email automation system."""
    print("\n📧 ECH0 Email Automation Demo\n")

    automation = ECH0_EmailAutomation()

    # Test customer data
    test_customers = [
        {
            'email': 'test1@example.com',
            'first_name': 'Alice',
            'industry': 'aerospace',
            'trial_created_date': (datetime.now() - timedelta(days=3)).isoformat(),
            'api_requests_count': 25,
            'current_tool': 'COMSOL',
            'day3_email_sent': False
        },
        {
            'email': 'test2@example.com',
            'first_name': 'Bob',
            'industry': 'materials science',
            'trial_created_date': (datetime.now() - timedelta(days=12)).isoformat(),
            'api_requests_count': 150,
            'tier': 'professional',
            'day12_email_sent': False
        },
        {
            'email': 'test3@example.com',
            'first_name': 'Carol',
            'industry': 'automotive',
            'trial_created_date': (datetime.now() - timedelta(days=7)).isoformat(),
            'api_requests_count': 5,
            'day7_email_sent': False
        }
    ]

    # Check triggers and send emails (dry run)
    print("Checking automation triggers for 3 test customers...\n")
    emails_sent = automation.check_triggers(test_customers, dry_run=True)

    print(f"\n✅ {len(emails_sent)} emails would be sent:")
    for email in emails_sent:
        print(f"  - {email['customer']}: {email['template']}")

    # Demo: Send single template
    print("\n" + "="*70)
    print("Demo: Sending cold outreach email")
    print("="*70 + "\n")

    cold_email_data = {
        'email': 'prospect@example.com',
        'first_name': 'David',
        'industry': 'aerospace engineering',
        'youtube_proof_link': 'https://youtube.com/watch?v=XXXXX'
    }

    automation.send_template(
        'cold_outreach_materials_scientists',
        cold_email_data,
        dry_run=True  # Set to False to actually send
    )


def launch_automation(customers_file: str = "customers.json", dry_run: bool = True):
    """
    Launch email automation for real customer data.

    Args:
        customers_file: Path to JSON file with customer data
        dry_run: If True, don't actually send emails (default True for safety)
    """
    print(f"\n🚀 Launching ECH0 Email Automation")
    print(f"Dry Run: {dry_run}")
    print(f"{'='*70}\n")

    automation = ECH0_EmailAutomation()

    # Load customers from file
    try:
        with open(customers_file, 'r') as f:
            customers = json.load(f)
    except FileNotFoundError:
        print(f"❌ Customers file not found: {customers_file}")
        print("Creating sample customers file...")

        sample_customers = [
            {
                'email': 'customer1@example.com',
                'first_name': 'John',
                'industry': 'aerospace',
                'trial_created_date': (datetime.now() - timedelta(days=3)).isoformat(),
                'api_requests_count': 50
            }
        ]

        with open(customers_file, 'w') as f:
            json.dump(sample_customers, f, indent=2)

        print(f"✅ Sample file created: {customers_file}")
        print("Edit this file with your real customer data, then run again.")
        return

    # Check triggers and send
    emails_sent = automation.check_triggers(customers, dry_run=dry_run)

    print(f"\n✅ Email automation complete!")
    print(f"Emails sent: {len(emails_sent)}")

    # Save log
    automation.save_log()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--launch":
        # Launch for real (still dry_run=True by default)
        dry_run = '--live' not in sys.argv
        launch_automation(dry_run=dry_run)
    else:
        # Demo mode
        demo_email_automation()

    print("\n💡 To launch automation:")
    print("  python ech0_email_automation.py --launch         # Dry run")
    print("  python ech0_email_automation.py --launch --live  # Actually send\n")
