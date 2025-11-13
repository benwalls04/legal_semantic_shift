import requests, json, time, os, argparse
from dotenv import load_dotenv  

load_dotenv()

API_URL = "https://www.courtlistener.com/api/rest/v4/opinions/"
TOKEN = os.getenv("COURTLISTENER_API_TOKEN")
HEADERS = {"Authorization": f"Token {TOKEN}"}

def fetch_opinions(decade, model_size, court="scotus"):
    start_date = f"{decade}-01-01"
    end_date = f"{decade + 9}-12-31"

    model_sizes = {
        "small": 100,
        "medium": 1000, 
        "large": 10000,
    }

    n_examples = model_sizes[model_size]
    outfile = f"../../data/inputs/opinions-{decade}-{model_size}.jsonl"

    os.makedirs(os.path.dirname(outfile), exist_ok=True)


    url = (f"{API_URL}?cluster__docket__court={court}"
           f"&date_filed__gte={start_date}&date_filed__lte={end_date}"
           f"&fields=id,date_filed,absolute_url,plain_text"
           f"&order_by=id")

    session = requests.Session()
    session.headers.update(HEADERS)

    count = 0
    with open(outfile, "w", encoding="utf-8") as f:
        while url and count < n_examples:
            resp = session.get(url)
            if resp.status_code != 200:
                print("Error:", resp.status_code, resp.text)
                break

            data = resp.json()

            for item in data.get("results", []):

                text = item.get("plain_text", "")
                if not text.strip():
                    continue

                json.dump(item, f)
                f.write("\n")
                count += 1
                if count >= n_examples:
                    print(f"✅ Collected {count} opinions. Stopping early.")
                    return

            url = data.get("next")
            if url:
                time.sleep(0.3) 
            else:
                break

    print(f"✅ Finished. Saved {count} opinions to {outfile}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch court opinions for a specific decade")
    parser.add_argument("--decade", type=int, required=True,
                       help="Decade to query (e.g., 1900, 1910, 1920)")
    parser.add_argument("--size", type=str, choices=["small", "medium", "large"], 
                       default="small",
                       help="Dataset size: small (200), medium (1k), large (50k). Default: small")


    args = parser.parse_args()
    fetch_opinions(args.decade, args.size)
