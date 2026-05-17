import requests
import time
import pandas as pd


HEADERS = {
    "User-Agent": "gvip-predictor bencheng18@gmail.com"
}


def get_fund_filings(cik: dir) -> dict:
    """
    Given a fund's CIK number, fetch all their SEC filings metadata.
    CIK is the unique ID the SEC assigns to every filing entity.
    """
    cik_padded = cik.zfill(10) # SEC requires 10-digit CIK
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"

    response = requests.get(url, headers=HEADERS)
    time.sleep(0.1)

    if response.status_code == 200:
        return response.json()
    else: 
        print(f"Failed to fetch CIK {cik}: {response.status_code}")
        return {}


def get_filing_index(accession_number: str, cik: str) -> dict:
    """
    Given an accession number, fetch the filing index to find
    the actual XML holdings document URL.
    """
    # Format accession number for URL (remove dashes)
    acc_formatted = accession_number.replace("-", "")
    cik_padded = cik.zfill(10)

    index_url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    
    doc_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_formatted}/{accession_number}-index.htm"
    
    response = requests.get(doc_url, headers=HEADERS)
    time.sleep(0.1)
    
    print("Index URL:", doc_url)
    print("Status:", response.status_code)
    return doc_url


def get_holdings_xml_url(accession_number: str, cik: str) -> str:
    """
    Parse the filing index page to find the URL of the 
    holdings XML document (the information table).
    """
    acc_formatted = accession_number.replace("-", "")
    cik_int = int(cik)
    
    index_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_formatted}/{accession_number}-index.htm"
    
    response = requests.get(index_url, headers=HEADERS)
    time.sleep(0.1)

    # Fina all XML files in the index page
    from lxml import etree
    content = response.text

    # Look for XML filenames in the HTML
    import re
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
    from lxml import etree

    response = requests.get(xml_url, headers=HEADERS)
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


if __name__ == "__main__":
    result = get_fund_filings("1067983")
    filings = result["filings"]["recent"]
    
    import pandas as pd
    df = pd.DataFrame(filings)
    df_13f = df[df["form"] == "13F-HR"]
    
    latest = df_13f.iloc[0]
    accession = latest["accessionNumber"]
    
    xml_url = get_holdings_xml_url(accession, "1067983")
    holdings_df = parse_holdings_xml(xml_url)

    # Aggregate duplicate CUSIPs
    holdings_df = holdings_df.groupby(["cusip", "name"]).agg(
        value=("value", "sum"),
        shares=("shares", "sum")
    ).reset_index()
    
    # Sort by value descending
    holdings_df = holdings_df.sort_values("value", ascending=False)
    
    print(holdings_df.head(10))
    print("Unique holdings:", len(holdings_df))