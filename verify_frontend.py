from playwright.sync_api import sync_playwright

def verify():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto("file:///app/website/qulab.aios.is/index.html")
        page.wait_for_selector('.lab-button.active')

        aria_label = page.locator('.prompt-bar input').get_attribute('aria-label')
        aria_current = page.locator('.lab-button.active').get_attribute('aria-current')

        print(f"aria-label on prompt-bar input: {aria_label}")
        print(f"aria-current on active lab-button: {aria_current}")

        assert aria_label == 'Ask for a combo', f"Expected 'Ask for a combo', got {aria_label}"
        assert aria_current == 'true', f"Expected 'true', got {aria_current}"

        page.screenshot(path="screenshot.png")
        print("Screenshot saved to screenshot.png")
        browser.close()

if __name__ == "__main__":
    verify()
