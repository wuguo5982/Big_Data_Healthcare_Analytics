"""PubMed E-utilities: public metadata/abstract retrieval; not a clinical guideline validator."""
import os
import requests
from xml.etree import ElementTree as ET
from crewai.tools import tool

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

@tool("Search PubMed for blood cancer publications")
def search_pubmed(query: str) -> str:
    """Search PubMed and return up to five papers with PMID, title, year, and abstract excerpt. Query must not contain patient identifiers."""
    if len(query) > 300:
        return "Query too long. Use a short non-identifying scientific search query."
    params = {"db": "pubmed", "term": query, "retmax": 5, "sort": "relevance", "retmode": "json", "tool": "blood_cancer_research"}
    if os.getenv("NCBI_EMAIL"):
        params["email"] = os.environ["NCBI_EMAIL"]
    try:
        resp = requests.get(BASE + "esearch.fcgi", params=params, timeout=20)
        resp.raise_for_status()
        ids = resp.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return "No PubMed records found for this query."
        response = requests.get(BASE + "efetch.fcgi", params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml", "tool": "blood_cancer_research"}, timeout=25)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        out = []
        for article in root.findall(".//PubmedArticle"):
            pmid = article.findtext(".//MedlineCitation/PMID", default="")
            title_node = article.find(".//ArticleTitle")
            title = " ".join("".join(title_node.itertext()).split()) if title_node is not None else "Untitled"
            year = article.findtext(".//PubDate/Year") or article.findtext(".//DateCompleted/Year") or "unknown"
            abstracts = article.findall(".//Abstract/AbstractText")
            abstract = " ".join(" ".join("".join(n.itertext()).split()) for n in abstracts)[:900]
            out.append(f"PMID: {pmid} | Year: {year} | Title: {title} | URL: https://pubmed.ncbi.nlm.nih.gov/{pmid}/ | Abstract excerpt: {abstract or 'Not available'}")
        return "\n\n".join(out)
    except (requests.RequestException, ValueError, ET.ParseError) as exc:
        return f"PubMed retrieval failed ({type(exc).__name__}); report missing evidence, do not invent citations."
