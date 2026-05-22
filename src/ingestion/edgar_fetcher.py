import requests
import time
import re
import pandas as pd
from lxml import etree
import yaml
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

HEADERS = {"User-Agent": config["edgar"]["user_agent"]}
TIMEOUT = config["pipeline"]["request_timeout"]


def safe_get(url: str, retries = 3) -> requests.Response:
    """
    GET request with retry and exponential backoff.
    Handles SEC rate limiting gracefully.
    """
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            if response.status_code == 429: # rate limited
                wait = 2 ** attempt
                print(f"Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            time.sleep(0.1)
            return response
        except requests.exceptions.Timeout:
            print(f"Timeout on attempt {attempt + 1}: {url}")
            time.sleep(2 ** attempt)
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            time.sleep(2 ** attempt)
    return None


def get_fund_filings(cik: dir) -> dict:
    """
    Given a fund's CIK number, fetch all their SEC filings metadata.
    Handles pagination for funds with large filing histories.
    """
    cik_padded = cik.zfill(10) # SEC requires 10-digit CIK
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"

    response = safe_get(url)
    if response is None or response.status_code != 200:
        print(f"Failed to fetch CIK {cik}: {response.status_code if response else 'No response'}")
        return {}
    time.sleep(0.1)

    data = response.json()

    # Check for additional paginated filing files
    if "files" in data.get("filings", {}):
        for file_info in data["filings"]["files"]:
            file_url = f"https://data.sec.gov/submissions/{file_info['name']}"
            file_response = safe_get(file_url)
            if file_response and file_response.status_code == 200:
                extra = file_response.json()
                # Merge extra filings into recent
                for key in data["filings"]["recent"]:
                    if key in extra:
                        data["filings"]["recent"][key].extend(extra[key])
    
    return data



def get_holdings_xml_url(accession_number: str, cik: str) -> str:
    """
    Parse the filing index page to find the URL of the 
    holdings XML document (the information table).
    """
    acc_formatted = accession_number.replace("-", "")
    cik_int = int(cik)
    
    index_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_formatted}/{accession_number}-index.htm"
    
    response = safe_get(index_url)
    if response is None or response.status_code != 200:
        return {}
    time.sleep(0.1)

    # Fina all XML files in the index page
    content = response.text

    # Look for XML filenames in the HTML
    xml_files = re.findall(r'href="([^"]+\.xml)"', content)
    
    # Filter out primary_doc.xml — we want the holdings table
    holdings_xml = [f for f in xml_files if "primary_doc" not in f]

    if holdings_xml:
        filename = holdings_xml[0].split("/")[-1]
        full_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_formatted}/{filename}"
        return full_url
    else:
        print("No holdings XML found")
        return ""

  
def parse_holdings_xml(xml_url: str) -> pd.DataFrame:
    """
    Fetch and parse the holdings XML file into a clean DataFrame.
    Each row = one stock holding.
    """

    response = safe_get(xml_url)
    if response is None or response.status_code != 200:
        return {}
    time.sleep(0.1)

    # Parse XML
    root = etree.fromstring(response.content)

    # XML has namespaces we need to handle
    namespace = {"ns": "http://www.sec.gov/edgar/document/thirteenf/informationtable"}
    
    holdings = []
    for info in root.findall("ns:infoTable", namespace):
        holding = {
            "name": info.findtext("ns:nameOfIssuer", namespaces=namespace),
            "cusip": info.findtext("ns:cusip", namespaces=namespace),
            "value": info.findtext("ns:value", namespaces=namespace),
            "shares": info.find("ns:shrsOrPrnAmt", namespace).findtext("ns:sshPrnamt", namespaces=namespace),
        }
        holdings.append(holding)
    
    df = pd.DataFrame(holdings)
    df["value"] = pd.to_numeric(df["value"])
    df["shares"] = pd.to_numeric(df["shares"])
    
    return df


def parse_cover_page(accession_number: str, cik: str) -> dict:
    """
    Fetch and parse the cover page XML to get portfolio summary.
    Much faster than fetching full holdings — used for pre-filtering.
    """
    acc_formatted = accession_number.replace("-", "")
    cik_int = int(cik)
    
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_formatted}/primary_doc.xml"
    
    response = safe_get(url)
    if response is None or response.status_code != 200:
        return {}
    time.sleep(0.1)
    
    if response.status_code != 200:
        return {}
    
    root = etree.fromstring(response.content)
    
    NS = "http://www.sec.gov/edgar/thirteenffiler"
    
    def find_text(tag: str) -> str:
        el = root.find(f".//{{{NS}}}{tag}")
        return el.text if el is not None else None
    
    return {
        "value_total": find_text("tableValueTotal"),
        "entry_total": find_text("tableEntryTotal"),
        "period_of_report": find_text("periodOfReport"),
        "included_managers_count": find_text("otherIncludedManagersCount"),
        "is_confidential_omitted": find_text("isConfidentialOmitted")
    }





if __name__ == "__main__":
    result = get_fund_filings("1067983")
    filings = result["filings"]["recent"]
    
    import pandas as pd
    df = pd.DataFrame(filings)
    df_13f = df[df["form"] == "13F-HR"]
    
    latest = df_13f.iloc[0]
    accession = latest["accessionNumber"]

    cover = parse_cover_page(accession, "1067983")
    print("Cover page data:")
    for k, v in cover.items():
        print(f"  {k}: {v}")