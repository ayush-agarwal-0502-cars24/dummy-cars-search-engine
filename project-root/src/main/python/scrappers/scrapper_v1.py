from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import pandas as pd

# ✅ Use webdriver-manager to get ChromeDriver
options = Options()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

# Automatically download and set up ChromeDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

print("🔍 Opening the Cars24 site...")
driver.get("https://www.cars24.com/buy-used-car/")

# Let listings load by scrolling
for i in range(20):
    print(f"🔄 Scrolling... ({i + 1}/20)")
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)

print("✅ Done scrolling. Parsing cars now...")

soup = BeautifulSoup(driver.page_source, 'html.parser')
car_cards = soup.select("a.styles_carCardWrapper__sXLIp")

print(f"🚗 Found {len(car_cards)} car cards")

data = []
for i, card in enumerate(car_cards, start=1):
    try:
        title = card.select_one("span.fkorgQ").text.strip()
        variant = card.select_one("span.iaEepT").text.strip()
        details = card.select("p.ehFgJI")
        price = card.select_one("p.lgDpJS").text.strip()
        location = card.select_one("p.jDkQaz").text.strip()
        image_url = card.select_one("img.shrinkOnTouch")["src"]
        car_url = "https://www.cars24.com" + card["href"]

        data.append({
            "title": title,
            "variant": variant,
            "kms": details[0].text.strip() if len(details) > 0 else "",
            "fuel": details[1].text.strip() if len(details) > 1 else "",
            "transmission": details[2].text.strip() if len(details) > 2 else "",
            "owner": details[3].text.strip() if len(details) > 3 else "",
            "price": price,
            "location": location,
            "image_url": image_url,
            "car_url": car_url
        })

        print(f"✅ Parsed car {i}: {title}")

    except Exception as e:
        print(f"❌ Error parsing card {i}: {e}")

driver.quit()

# Save to CSV
df = pd.DataFrame(data)
df.to_csv("cars24_listings.csv", index=False)
print("💾 Saved data to cars24_listings.csv")