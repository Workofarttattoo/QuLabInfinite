"""
Stripe integration for QuLab Billing.
"""
from __future__ import annotations

import os

import stripe

from qulab_billing.pricing import CREDIT_BUNDLES


def _get_stripe_key() -> str:
    key = os.environ.get("STRIPE_SECRET_KEY", "")
    if not key:
        raise RuntimeError("STRIPE_SECRET_KEY not configured")
    return key


def get_bundle(index: int) -> dict:
    """Return the credit bundle dict at the given index."""
    if index < 0 or index >= len(CREDIT_BUNDLES):
        raise ValueError(f"Invalid bundle index: {index}. Valid range: 0-{len(CREDIT_BUNDLES) - 1}")
    return CREDIT_BUNDLES[index]


def create_stripe_customer(email: str) -> str:
    """
    Create a Stripe customer for the given email.

    Returns the Stripe customer ID string.
    """
    stripe.api_key = _get_stripe_key()
    customer = stripe.Customer.create(email=email)
    return customer.id  # type: ignore[return-value]


def create_checkout_session(
    stripe_customer_id: str,
    bundle_index: int,
    success_url: str,
    cancel_url: str,
) -> str:
    """
    Create a Stripe Checkout Session for the chosen credit bundle.

    Returns the checkout session URL.
    """
    stripe.api_key = _get_stripe_key()
    bundle = get_bundle(bundle_index)

    session = stripe.checkout.Session.create(
        customer=stripe_customer_id,
        payment_method_types=["card"],
        line_items=[
            {
                "price_data": {
                    "currency": "usd",
                    "unit_amount": bundle["price_cents"],
                    "product_data": {
                        "name": f"QuLab Credits — {bundle['label']}",
                        "description": bundle["description"],
                    },
                },
                "quantity": 1,
            }
        ],
        mode="payment",
        success_url=success_url,
        cancel_url=cancel_url,
        metadata={
            "user_credit_bundle": str(bundle_index),
            "purchase_type": "qulab_credits",
        },
    )
    return session.url  # type: ignore[return-value]


def construct_webhook_event(payload: bytes, sig_header: str) -> stripe.Event:
    """
    Verify and parse a Stripe webhook payload.

    Reads STRIPE_WEBHOOK_SECRET from environment.
    Raises stripe.error.SignatureVerificationError on invalid signature.
    """
    webhook_secret = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
    if not webhook_secret:
        raise RuntimeError("STRIPE_WEBHOOK_SECRET not configured")

    stripe.api_key = _get_stripe_key()
    event = stripe.Webhook.construct_event(
        payload=payload,
        sig_header=sig_header,
        secret=webhook_secret,
    )
    return event  # type: ignore[return-value]
