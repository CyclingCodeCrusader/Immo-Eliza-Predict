
with open("scripts/WebScraper.py", "r") as f:
    WebScraper = f.read()

exec(WebScraper)

with open("scripts/DataCleaner.py", "r") as f:
    DataCleaner = f.read()

exec(DataCleaner)


with open("scripts/Analyzer.py", "r") as f:
    Analyzer = f.read()

exec(Analyzer)

with open("scripts/Modeller.py", "r") as f:
    Modeller = f.read()

exec(Modeller)

