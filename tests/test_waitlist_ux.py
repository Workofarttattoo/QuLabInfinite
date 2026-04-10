import os
import re

def test_waitlist_ux_improvements():
    filepath = os.path.join(os.path.dirname(__file__), "..", "website", "qulab.aios.is", "index.html")
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Verify aria-label on email input
    assert 'id="emailInput"' in content
    assert 'aria-label="Email address"' in content

    # Verify artificial delay for perceived reliability
    assert "button.textContent = 'Requesting...';" in content
    assert "button.disabled = true;" in content
    assert "}, 600);" in content
