import asyncio
from playwright.async_api import async_playwright
import os

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        file_path = f"file://{os.path.abspath('index.html')}"
        await page.goto(file_path)

        # Test focus on search bar
        await page.focus('#labSearch')
        await page.screenshot(path="screenshot_search_focus.png")

        # Test focus on prompt bar input
        await page.focus('#promptInput')
        await page.screenshot(path="screenshot_prompt_focus.png")

        # Test focus on send button
        await page.focus('.prompt-bar button')
        await page.screenshot(path="screenshot_button_focus.png")

        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
