import requests, json, time, os, argparse, re, random
from html.parser import HTMLParser
from dotenv import load_dotenv  

load_dotenv()

API_URL = "https://www.courtlistener.com/api/rest/v4/opinions/"
CLUSTER_API_URL = "https://www.courtlistener.com/api/rest/v4/clusters/"
TOKEN = os.getenv("COURTLISTENER_API_TOKEN")
HEADERS = {"Authorization": f"Token {TOKEN}"}

class HTMLTextExtractor(HTMLParser):
    """
    HTML parser to extract text content from HTML, removing all tags.
    """
    def __init__(self):
        super().__init__()
        self.text = []
        self.ignore_tags = {'script', 'style'}  # Tags to ignore content from
        self.current_tag = None
    
    def handle_starttag(self, tag, attrs):
        self.current_tag = tag.lower()
        # Add newline for block elements
        if tag.lower() in {'p', 'div', 'br', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'tr'}:
            self.text.append('\n')
    
    def handle_endtag(self, tag):
        self.current_tag = None
        # Add newline for block elements
        if tag.lower() in {'p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'}:
            self.text.append('\n')
    
    def handle_data(self, data):
        # Only add text if not inside ignored tags
        if self.current_tag not in self.ignore_tags:
            self.text.append(data)
    
    def get_text(self):
        """Return the extracted text, cleaned up."""
        text = ''.join(self.text)
        # Clean up multiple whitespace/newlines
        text = re.sub(r'\n\s*\n+', '\n\n', text)  # Multiple newlines to double newline
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single space
        return text.strip()

def extract_text_from_html(html_content):
    """
    Extract plain text from HTML content by removing all tags.
    Returns the cleaned text.
    """
    if not html_content or not isinstance(html_content, str):
        return ""
    
    parser = HTMLTextExtractor()
    try:
        parser.feed(html_content)
        return parser.get_text()
    except Exception as e:
        print(f"  Warning: Error parsing HTML: {e}")
        return ""

def map_clusters_by_decade(court="scotus"):
    """
    Fetch all clusters, parse years from other_dates field, and map cluster IDs to decades.
    Writes a JSON file mapping decades to lists of cluster IDs.
    """
    url = f"{CLUSTER_API_URL}?docket__court={court}&order_by=id"
    output_file = "../../data/decade_to_clusters.json"
    
    print(f"\nQuery URL: {url}")
    print(f"Fetching all clusters for court: {court}")
    print(f"Parsing years from other_dates field and mapping to decades...\n")
    
    session = requests.Session()
    session.headers.update(HEADERS)
    
    # Dictionary to map decades to cluster IDs
    decade_to_clusters = {}
    
    # Regex pattern to find 4-digit years (reasonable range: 1800-2100)
    year_pattern = re.compile(r'\b(1[89]\d{2}|20[0-1]\d|2100)\b')
    
    count = 0
    clusters_with_year = 0
    clusters_without_year = 0
    
    while url:
        resp = session.get(url)
        if resp.status_code != 200:
            print("Error:", resp.status_code, resp.text)
            break
        
        data = resp.json()
        
        for cluster in data.get("results", []):
            cluster_id = cluster.get("id")
            other_dates = cluster.get("other_dates", "")
            
            if cluster_id:
                # Find all 4-digit years in other_dates
                years = year_pattern.findall(other_dates)
                
                if years:
                    # Use the first year found (naive approach as requested)
                    year = int(years[0])
                    # Map to decade (e.g., 1904 -> 1900, 2010 -> 2010)
                    decade = (year // 10) * 10
                    
                    # Initialize decade list if it doesn't exist
                    if decade not in decade_to_clusters:
                        decade_to_clusters[decade] = []
                    
                    decade_to_clusters[decade].append(cluster_id)
                    clusters_with_year += 1
                else:
                    clusters_without_year += 1
                
                count += 1
                
                # Progress update every 1000 clusters
                if count % 1000 == 0:
                    print(f"Processed {count} clusters... (with year: {clusters_with_year}, without: {clusters_without_year})")
        
        url = data.get("next")
        if url:
            time.sleep(0.3)
        else:
            break
    
    # Sort decades and cluster IDs for consistency
    sorted_decades = sorted(decade_to_clusters.keys())
    for decade in sorted_decades:
        decade_to_clusters[decade].sort()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Write to JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(decade_to_clusters, f, indent=2)
    
    print(f"\n✅ Finished processing {count} clusters.")
    print(f"   Clusters with year found: {clusters_with_year}")
    print(f"   Clusters without year: {clusters_without_year}")
    print(f"   Decades found: {sorted_decades}")
    print(f"   Output written to: {output_file}")

def fetch_opinions(decade, model_size, court="scotus", mapping_file="../../data/decade_to_clusters.json"):
    """
    Fetch opinions for a specific decade by:
    1. Loading the cluster mapping file
    2. Getting all cluster IDs for the decade
    3. Fetching all opinions from those clusters
    4. Randomly sampling until model_size limit is reached
    """
    model_sizes = {
        "small": 100,
        "medium": 1000, 
        "large": 10000,
    }

    n_examples = model_sizes[model_size]
    outfile = f"../../data/inputs/opinions-{decade}-{model_size}.jsonl"

    os.makedirs(os.path.dirname(outfile), exist_ok=True) 

    # Load the cluster mapping
    print(f"\nLoading cluster mapping from: {mapping_file}")
    try:
        with open(mapping_file, "r", encoding="utf-8") as f:
            decade_to_clusters = json.load(f)
    except FileNotFoundError:
        print(f"Error: Mapping file not found: {mapping_file}")
        print("Please run --map-decades first to generate the mapping file.")
        return
    
    # Get cluster IDs for this decade
    cluster_ids = decade_to_clusters.get(str(decade), [])
    if not cluster_ids:
        print(f"Error: No clusters found for decade {decade}")
        print(f"Available decades: {list(decade_to_clusters.keys())}")
        return
    
    print(f"Found {len(cluster_ids)} clusters for decade {decade}")
    print(f"Fetching opinions from all clusters, then sampling {n_examples} opinions...\n")

    session = requests.Session()
    session.headers.update(HEADERS)

    # Collect all opinions from all clusters
    all_opinions = []
    total_opinions_found = 0
    opinions_with_text = 0
    opinions_without_text = 0
    opinions_from_html = 0
    clusters_with_opinions = 0
    clusters_without_opinions = 0
    
    for idx, cluster_id in enumerate(cluster_ids):
        # Stop if we've collected enough opinions
        if len(all_opinions) >= n_examples:
            print(f"\nReached target of {n_examples} opinions. Stopping early.")
            break
        
        # Fetch all opinions for this cluster (get all fields, not just limited set)
        url = f"{API_URL}?cluster={cluster_id}"
        
        cluster_opinions_total = 0
        cluster_opinions_with_text = 0
        cluster_opinions_without_text = 0
        opinions_without_text_list = []  # Store opinions without text for printing
        
        while url:
            # Stop if we've collected enough opinions
            if len(all_opinions) >= n_examples:
                break
            
            resp = session.get(url)
            if resp.status_code != 200:
                print(f"Error fetching cluster {cluster_id}: {resp.status_code} - {resp.text}")
                break
            
            data = resp.json()
            
            for item in data.get("results", []):
                # Stop if we've collected enough opinions
                if len(all_opinions) >= n_examples:
                    break
                
                cluster_opinions_total += 1
                total_opinions_found += 1
                
                text = item.get("plain_text", "")
                
                # If no plain_text but html exists, extract text from HTML
                if not text.strip() and item.get("html"):
                    html_content = item.get("html", "")
                    extracted_text = extract_text_from_html(html_content)
                    if extracted_text.strip():
                        # Add the extracted text as plain_text
                        item["plain_text"] = extracted_text
                        text = extracted_text
                        opinions_from_html += 1
                        print(f"  Extracted text from HTML for opinion ID {item.get('id', 'N/A')} ({len(extracted_text)} chars)")
                
                if text.strip():  # Only include opinions with non-empty text
                    all_opinions.append(item)
                    cluster_opinions_with_text += 1
                    opinions_with_text += 1
                    
                    # Stop if we've reached the limit
                    if len(all_opinions) >= n_examples:
                        break
                else:
                    cluster_opinions_without_text += 1
                    opinions_without_text += 1
                    opinions_without_text_list.append(item)
            
            # Break out of while loop if we've reached the limit
            if len(all_opinions) >= n_examples:
                break
            
            url = data.get("next")
            if url:
                time.sleep(0.3)
            else:
                break
        
        # Track cluster statistics
        if cluster_opinions_total > 0:
            clusters_with_opinions += 1
            if cluster_opinions_without_text > 0:
                # Print info for clusters that have opinions but no text
                print(f"\nCluster {cluster_id}: {cluster_opinions_total} opinions found - "
                      f"{cluster_opinions_with_text} with text, {cluster_opinions_without_text} without text")
                # Print all fields for opinions without text
                for i, opinion in enumerate(opinions_without_text_list):
                    print(f"\n  Opinion without plain_text ({i+1}/{len(opinions_without_text_list)}):")
                    #print_opinion_fields(opinion)
        
        # Break out of cluster loop if we've reached the limit
        if len(all_opinions) >= n_examples:
            break
        
        # Progress update every 10 clusters
        if (idx + 1) % 10 == 0:
            print(f"\nProcessed {idx + 1}/{len(cluster_ids)} clusters: "
                  f"{total_opinions_found} total opinions found, "
                  f"{opinions_with_text} with text, "
                  f"{opinions_without_text} without text, "
                  f"{opinions_from_html} extracted from HTML")
        
        time.sleep(0.2)  # Rate limiting between clusters
    
    print(f"\n{'='*60}")
    print(f"Summary Statistics:")
    print(f"  Total clusters checked: {min(idx + 1, len(cluster_ids))}")
    print(f"  Clusters with opinions: {clusters_with_opinions}")
    print(f"  Clusters without opinions: {clusters_without_opinions}")
    print(f"  Total opinions found: {total_opinions_found}")
    print(f"  Opinions with plain_text: {opinions_with_text}")
    print(f"  Opinions extracted from HTML: {opinions_from_html}")
    print(f"  Opinions without plain_text: {opinions_without_text}")
    print(f"{'='*60}")
    print(f"\nCollected {len(all_opinions)} opinions with text")
    
    # Randomly sample opinions (if we collected more than needed)
    if len(all_opinions) > n_examples:
        print(f"Randomly sampling {n_examples} from {len(all_opinions)} collected opinions...")
        sampled_opinions = random.sample(all_opinions, n_examples)
    else:
        sampled_opinions = all_opinions
        if len(all_opinions) < n_examples:
            print(f"Warning: Only {len(all_opinions)} opinions available, less than requested {n_examples}")
    
    # Write to output file
    with open(outfile, "w", encoding="utf-8") as f:
        for item in sampled_opinions:
            json.dump(item, f)
            f.write("\n")
    
    print(f"✅ Finished. Saved {len(sampled_opinions)} opinions to {outfile}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch court opinions or print cluster dates")
    parser.add_argument("--decade", type=int, default=None,
                       help="Decade to query (e.g., 1900, 1910, 1920)")
    parser.add_argument("--size", type=str, choices=["small", "medium", "large"], 
                       default="small",
                       help="Dataset size: small (200), medium (1k), large (50k). Default: small")
    parser.add_argument("--map-decades", action="store_true",
                       help="Map all clusters to decades based on years in other_dates field")

    args = parser.parse_args()
    
    if args.map_decades:
        map_clusters_by_decade(court="scotus")
    else:
        if args.decade is None:
            parser.error("--decade is required when not using --cluster-dates or --map-decades")
        fetch_opinions(args.decade, args.size)
