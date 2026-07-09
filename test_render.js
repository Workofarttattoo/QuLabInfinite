const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto('file:///app/website/qulab.aios.is/index.html');
  const activeClassCount = await page.evaluate(() => document.querySelectorAll('.lab-button.active').length);
  const ariaCurrentCount = await page.evaluate(() => document.querySelectorAll('.lab-button[aria-current="true"]').length);
  console.log(`Active buttons: ${activeClassCount}`);
  console.log(`Aria-current buttons: ${ariaCurrentCount}`);
  await browser.close();
})();
