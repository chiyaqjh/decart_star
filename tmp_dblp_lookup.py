import re
import ssl
import urllib.parse
import urllib.request

ctx = ssl._create_unverified_context()
queries = [
    ("delphi", "Delphi A Cryptographic Inference Service for Neural Networks"),
    (
        "offline_owner",
        "Efficient k-NN query over encrypted data in cloud with limited key-disclosure and offline data owner",
    ),
    (
        "query_access_control",
        "Access Control Enforcement on Query-Aware Encrypted Cloud Databases",
    ),
]

for key, query in queries:
    url = "https://dblp.org/search?q=" + urllib.parse.quote(query)
    print(f"--- {key} ---")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20, context=ctx) as response:
            html = response.read().decode("utf-8", errors="ignore")
        match = re.search(r"https://dblp.org/rec/[^\"\s<>]+", html)
        if match:
            print(match.group(0))
        title_match = re.search(r"<title>(.*?)</title>", html, re.I | re.S)
        if title_match:
            print("PAGE_TITLE:", re.sub(r"\s+", " ", title_match.group(1)).strip())
    except Exception as exc:
        print("ERR", exc)
