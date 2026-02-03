import requests
import json

def test_slm(company_name, symbol, title, content):
    api_url = "http://localhost:18000/api/ask"
    
    prompt = f"""You are a financial news classifier. Determine if this article is about {company_name} ({symbol}) the technology company.

Article Title: {title}
Article Content: {content[:300]}

IMPORTANT: Answer with ONLY the word "YES" or "NO" as the first word of your response.

- YES: The article discusses {company_name} the company, its products, business, or stock
- NO: The article uses similar words but is NOT about the company

Examples:
- "Intel launches new chip" → YES
- "Beauty intel: new makeup trends" → NO
- "Intelligence Committee meeting" → NO
- "Intel CEO announces..." → YES
- "Microsoft buys SwiftKey" → YES
- "Apple iPhone security flaw" → YES

Answer (YES or NO):"""

    print(f"--- Testing {symbol} ---")
    print(f"Title: {title}")
    response = requests.post(api_url, json={"question": prompt})
    if response.status_code == 200:
        answer = response.json().get("answer", "")
        print(f"SLM Answer: {answer}")
    else:
        print(f"Error: {response.status_code}")

# Test Microsoft
test_slm("Microsoft", "MSFT", "Microsoft buys keyboard software maker SwiftKey", "Microsoft has confirmed it is acquiring London-based keyboard software maker SwiftKey for $250 million...")
