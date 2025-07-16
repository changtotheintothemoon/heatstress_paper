import os
import time
import json
from datetime import datetime
from Bio import Entrez

# Read Entrez credentials from environment
ENTREZ_EMAIL = os.environ.get('ENTREZ_EMAIL', 'your.email@example.com')  # Default for demo
ENTREZ_API_KEY = os.environ.get('ENTREZ_API_KEY')
if not ENTREZ_EMAIL or ENTREZ_EMAIL == 'your.email@example.com':
    print("Warning: Please set ENTREZ_EMAIL environment variable with your actual email")
    print("Using default email for demonstration purposes")

Entrez.email = ENTREZ_EMAIL
if ENTREZ_API_KEY:
    Entrez.api_key = ENTREZ_API_KEY

# NCBI rate limit: <=3 requests/sec
REQUEST_DELAY = 0.34  # seconds

# Query construction - Updated keywords and sub-keywords
search_terms = (
    '"heat stress"'
    ' AND (broiler OR broilers)'
    ' AND (manipulation OR performance OR production)'
    ' AND (antioxidants OR vitamins OR "vitamin E" OR "vitamin C" OR probiotics'
    ' OR phytogenics OR rutin OR resveratrol OR curcumin'
    ' OR minerals OR sodium OR potassium OR selenium OR zinc OR manganese'
    ' OR "amino acids" OR citrulline OR glutamine OR arginine OR methionine OR tryptophan'
    ' OR betaine)'
)
start_year = 1995
end_year = datetime.now().year

query = f'({search_terms}) AND ("{start_year}"[PDAT] : "{end_year}"[PDAT])'

# Step 1: Search PubMed
print('Searching PubMed...')
handle = Entrez.esearch(
    db='pubmed',
    term=query,
    retmax=1000,
    usehistory='y',
    sort='relevance'
)
record = Entrez.read(handle)
handle.close()
time.sleep(REQUEST_DELAY)

pmids = record['IdList']
print(f'Found {len(pmids)} articles.')

# Step 2: Fetch details for each PMID (in batches)
results = []
batch_size = 100
for i in range(0, len(pmids), batch_size):
    batch_pmids = pmids[i:i+batch_size]
    print(f'Fetching records {i+1} to {i+len(batch_pmids)}...')
    handle = Entrez.efetch(
        db='pubmed',
        id=','.join(batch_pmids),
        rettype='medline',
        retmode='xml'
    )
    records = Entrez.read(handle)
    handle.close()
    time.sleep(REQUEST_DELAY)
    for article in records['PubmedArticle']:
        medline = article['MedlineCitation']
        article_data = medline['Article']
        # Title
        title = article_data.get('ArticleTitle', '')
        # Year
        year = ''
        if 'ArticleDate' in article_data and article_data['ArticleDate']:
            year = article_data['ArticleDate'][0].get('Year', '')
        elif 'Journal' in article_data and 'JournalIssue' in article_data['Journal']:
            year = article_data['Journal']['JournalIssue'].get('PubDate', {}).get('Year', '')
        # Abstract
        abstract = ''
        if 'Abstract' in article_data and 'AbstractText' in article_data['Abstract']:
            abstract = ' '.join(str(x) for x in article_data['Abstract']['AbstractText'])
        # Authors
        authors = []
        for author in article_data.get('AuthorList', []):
            if 'ForeName' in author and 'LastName' in author:
                authors.append(f"{author['ForeName']} {author['LastName']}")
            elif 'LastName' in author:
                authors.append(author['LastName'])
        results.append({
            'pmid': medline.get('PMID', ''),
            'title': title,
            'year': year,
            'abstract': abstract,
            'authors': authors
        })

# Step 3: Save to JSON
with open('heat_stress_poultry.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f'Saved {len(results)} records to heat_stress_poultry.json') 