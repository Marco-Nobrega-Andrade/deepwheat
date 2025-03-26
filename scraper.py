#!/usr/bin/env python3
"""
Google Scholar Scraper for Wheat Disease Research

This script scrapes Google Scholar for research papers on wheat diseases:
- Yellow rust (stripe rust)
- Brown rust (leaf rust)
- Septoria
- Mildew (powdery mildew)

It downloads up to 100 PDFs for each disease and organizes them into folders.
"""

import os
import time
import random
import requests
import argparse
import concurrent.futures
from bs4 import BeautifulSoup
from urllib.parse import urljoin, quote_plus, urlparse
import PyPDF2
import logging
import re
import datetime

# Set up logging - simple version with no timestamp or subdirectory
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scholar_scraper.log", mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.4 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
]

BASE_DELAY = 3  # Base delay between requests in seconds
JITTER = 2  # Random jitter to add to delay
MAX_RETRIES = 5  # Maximum number of retries for failed requests

# Additional sources to search for PDFs
ADDITIONAL_SOURCES = [
    {"name": "Core.ac.uk", "url": "https://core.ac.uk/search?q={query}"},
    {"name": "ScienceDirect", "url": "https://www.sciencedirect.com/search?qs={query}"},
    {"name": "PubAg", "url": "https://pubag.nal.usda.gov/catalog/search?q={query}"}
]

DISEASES = [
    {"name": "yellow_rust", "queries": [
        "wheat yellow rust detection filetype:pdf",
        "wheat stripe rust management filetype:pdf", 
        "wheat yellow rust puccinia striiformis treatment filetype:pdf",
        "wheat stripe rust control prevention filetype:pdf",
        "yellow rust wheat agriculture filetype:pdf"
    ]},
    {"name": "brown_rust", "queries": [
        "wheat brown rust detection filetype:pdf",
        "wheat leaf rust management filetype:pdf",
        "wheat puccinia triticina treatment filetype:pdf",
        "wheat brown rust control prevention filetype:pdf",
        "leaf rust wheat agriculture filetype:pdf"
    ]},
    {"name": "septoria", "queries": [
        "wheat septoria detection filetype:pdf",
        "wheat septoria tritici blotch management filetype:pdf",
        "wheat zymoseptoria tritici treatment filetype:pdf",
        "wheat septoria control prevention filetype:pdf",
        "septoria wheat agriculture filetype:pdf"
    ]},
    {"name": "mildew", "queries": [
        "wheat powdery mildew detection filetype:pdf",
        "wheat blumeria graminis management filetype:pdf",
        "wheat mildew treatment filetype:pdf",
        "wheat powdery mildew control prevention filetype:pdf",
        "mildew wheat agriculture filetype:pdf"
    ]}
]

PDF_COUNT_PER_DISEASE = 100
GOOGLE_SCHOLAR_URL = "https://scholar.google.com/scholar"

def get_random_user_agent():
    """Return a random user agent from the list"""
    return random.choice(USER_AGENTS)

def create_output_dirs():
    """Create output directories for each disease"""
    for disease in DISEASES:
        os.makedirs(disease["name"], exist_ok=True)
    
    # Create a directory for papers that couldn't be validated
    os.makedirs("unvalidated", exist_ok=True)

def random_delay():
    """Sleep for a random amount of time to avoid being blocked"""
    delay = BASE_DELAY + random.uniform(0, JITTER)
    logger.info(f"Sleeping for {delay:.2f} seconds")
    time.sleep(delay)

def is_valid_pdf(file_path):
    """Check if the downloaded file is a valid PDF"""
    try:
        with open(file_path, 'rb') as f:
            # Try to read the first 1024 bytes to check if it's a PDF
            header = f.read(1024)
            if not header.startswith(b'%PDF-'):
                return False
            
            # Try to parse with PyPDF2
            f.seek(0)
            pdf = PyPDF2.PdfReader(f)
            # Try to get the number of pages to validate
            num_pages = len(pdf.pages)
            return True
    except Exception as e:
        logger.warning(f"Invalid PDF: {e}")
        return False

def download_pdf(url, output_path):
    """
    Download a PDF file from the given URL
    Returns True if successful, False otherwise
    """
    headers = {'User-Agent': get_random_user_agent()}
    
    # Try multiple times in case of temporary failures
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, headers=headers, stream=True, timeout=30)
            
            # Check if the response is successful
            if not response.ok:
                logger.warning(f"Failed to download (Status: {response.status_code}): {url}")
                time.sleep(BASE_DELAY)
                continue
                
            # Check if the response is a PDF
            content_type = response.headers.get('Content-Type', '').lower()
            if 'application/pdf' not in content_type and not url.lower().endswith('.pdf'):
                # If direct URL isn't a PDF, try to find a PDF link on the page
                if 'text/html' in content_type:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Look for direct PDF links
                    pdf_links = soup.select('a[href$=".pdf"]')
                    if pdf_links:
                        pdf_url = urljoin(url, pdf_links[0]['href'])
                        logger.info(f"Found PDF link in HTML page: {pdf_url}")
                        return download_pdf(pdf_url, output_path)
                    
                    # Look for PDF links with buttons/text indicating PDFs
                    potential_pdf_links = []
                    for link in soup.select('a'):
                        href = link.get('href', '')
                        if not href:
                            continue
                        text = link.text.lower()
                        if (href.endswith('.pdf') or 
                            'pdf' in text or 
                            'download' in text or 
                            'full text' in text or
                            'article' in text):
                            potential_pdf_links.append(link)
                    
                    if potential_pdf_links:
                        for pdf_link in potential_pdf_links[:3]:  # Try the first 3 potential links
                            pdf_url = urljoin(url, pdf_link['href'])
                            logger.info(f"Trying potential PDF link: {pdf_url}")
                            if download_pdf(pdf_url, output_path):
                                return True
                
                # Check if we're on a publisher page that might have special PDF links
                if any(domain in url.lower() for domain in ['sciencedirect', 'springer', 'wiley', 'mdpi', 'tandfonline']):
                    logger.info(f"Detected publisher page, searching for publisher-specific PDF links")
                    soup = BeautifulSoup(response.content, 'html.parser')
                    # Check various publisher-specific PDF link patterns
                    publisher_selectors = [
                        'a.pdf-link', 'a.download-link', 'a[data-track-action="download pdf"]', 
                        'a.article-dl-pdf-link', 'a[title*="PDF"]', 'a[aria-label*="PDF"]'
                    ]
                    for selector in publisher_selectors:
                        links = soup.select(selector)
                        if links:
                            pdf_url = urljoin(url, links[0]['href'])
                            logger.info(f"Found publisher-specific PDF link: {pdf_url}")
                            if download_pdf(pdf_url, output_path):
                                return True
                
                logger.warning(f"URL does not point to a PDF and couldn't find PDF link: {url}")
                return False
            
            # Save the PDF
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # Validate the PDF
            if is_valid_pdf(output_path):
                logger.info(f"Successfully downloaded and validated: {output_path}")
                return True
            else:
                logger.warning(f"Downloaded file is not a valid PDF: {output_path}")
                # Move to unvalidated folder
                unvalidated_path = os.path.join("unvalidated", os.path.basename(output_path))
                os.rename(output_path, unvalidated_path)
                return False
                
        except Exception as e:
            logger.error(f"Error downloading {url} (Attempt {attempt+1}/{MAX_RETRIES}): {e}")
            time.sleep(BASE_DELAY)
            continue
    
    return False

def extract_pdf_links_from_page(soup):
    """Extract PDF links and metadata from Google Scholar search results"""
    results = []
    
    # Find all search result entries
    entries = soup.select('div.gs_r.gs_or.gs_scl')
    
    for entry in entries:
        try:
            # Extract title
            title_elem = entry.select_one('.gs_rt a')
            title = title_elem.text if title_elem else "Untitled"
            
            # Extract authors and year
            author_year_elem = entry.select_one('.gs_a')
            author_year_text = author_year_elem.text if author_year_elem else ""
            
            # Get first author
            author_match = re.search(r'^([^,]+)', author_year_text)
            author = author_match.group(1).strip() if author_match else "Unknown"
            
            # Get year
            year_match = re.search(r'(\d{4})', author_year_text)
            year = year_match.group(1) if year_match else "Unknown"
            
            # Check for PDF links
            link_elems = entry.select('.gs_or_ggsm a')
            pdf_link = None
            
            # First try to find direct PDF links in the sidebar
            for link in link_elems:
                href = link.get('href', '')
                if href.lower().endswith('.pdf') or '[PDF]' in link.text:
                    pdf_link = href
                    break
            
            # If no PDF link in sidebar, try to find in the bottom links
            if not pdf_link:
                bottom_links = entry.select('.gs_fl a')
                for link in bottom_links:
                    if '[PDF]' in link.text or 'PDF' in link.text:
                        pdf_link = link.get('href', '')
                        break
            
            # If no PDF link found yet, try to get the main article link
            if not pdf_link and title_elem and title_elem.has_attr('href'):
                pdf_link = title_elem['href']
            
            # If we found a PDF link or a potential source link
            if pdf_link:
                results.append({
                    'url': pdf_link if pdf_link.startswith('http') else f"https://scholar.google.com{pdf_link}",
                    'title': title,
                    'author': author,
                    'year': year
                })
        except Exception as e:
            logger.error(f"Error processing search result: {e}")
            continue
    
    return results

def scrape_google_scholar_page(query, page=0, exact_match=False):
    """
    Scrape a single page of Google Scholar results
    Returns a list of dictionaries with paper information and links
    """
    start = page * 10  # Google Scholar shows 10 results per page
    
    # If exact match is requested, enclose query in quotes
    if exact_match and not query.startswith('"') and not query.endswith('"'):
        formatted_query = f'"{query}"'
    else:
        formatted_query = query
    
    # Add filetype:pdf to query if not already present
    if "filetype:pdf" not in formatted_query:
        formatted_query += " filetype:pdf"
        
    params = {
        'q': formatted_query,
        'hl': 'en',
        'start': start,
        'as_vis': '0',  # Include citations
        'as_sdt': '0,33',  # Search in articles
        'as_ylo': '2000'  # Only include results from 2000 onwards for more relevant papers
    }
    
    # For troubleshooting
    full_url = f"{GOOGLE_SCHOLAR_URL}?{'&'.join([f'{k}={v}' for k, v in params.items()])}"
    logger.debug(f"Full URL: {full_url}")
    
    headers = {
        'User-Agent': get_random_user_agent(),
        'Accept': 'text/html,application/xhtml+xml,application/xml',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://scholar.google.com/',
        'DNT': '1'
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Getting page {page} for query: {formatted_query}")
            
            # Add a small delay before the first request to avoid appearing automated
            if page == 0 and attempt == 0:
                time.sleep(random.uniform(1, 3))
                
            response = requests.get(GOOGLE_SCHOLAR_URL, params=params, headers=headers, timeout=30)
            
            if not response.ok:
                logger.warning(f"Failed to get search results (Status: {response.status_code})")
                if response.status_code == 429 or response.status_code == 503:
                    # Too many requests or service unavailable, wait longer
                    wait_time = BASE_DELAY * 2 * (attempt + 1)
                    logger.warning(f"Rate limited. Waiting {wait_time} seconds")
                    time.sleep(wait_time)
                    continue
                return [], False
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Check for CAPTCHA
            if 'Please show you&#39;re not a robot' in response.text or 'Our systems have detected unusual traffic' in response.text:
                logger.error("CAPTCHA detected. Try using a different IP or wait longer between requests")
                time.sleep(BASE_DELAY * 5)
                return [], False
            
            # Extract PDF links
            results = extract_pdf_links_from_page(soup)
            logger.info(f"Found {len(results)} potential PDF links on page {page}")
            
            # Check if there are more pages
            has_next_page = bool(soup.select('td.gs_ico a.gs_ico_nav_next')) or bool(soup.select('button.gs_btnPR'))
            
            return results, has_next_page
            
        except requests.exceptions.RequestException as e:
            # Handle network errors
            logger.error(f"Network error scraping Google Scholar page {page} (Attempt {attempt+1}/{MAX_RETRIES}): {e}")
            time.sleep(BASE_DELAY * (attempt + 1))
        except Exception as e:
            # Handle all other errors
            logger.error(f"Error scraping Google Scholar page {page} (Attempt {attempt+1}/{MAX_RETRIES}): {e}")
            time.sleep(BASE_DELAY * (attempt + 1))
    
    return [], False

def search_additional_sources(disease_name, queries, current_count, downloaded_urls):
    """
    Search additional sources for PDFs if Google Scholar didn't yield enough results
    """
    additional_count = current_count
    
    # Add more specialized agricultural databases
    specialized_sources = [
        {"name": "FAO", "url": "https://www.fao.org/search/en/?q={query}"},
        {"name": "CIMMYT", "url": "https://www.cimmyt.org/search/?q={query}"},
        {"name": "ICARDA", "url": "https://www.icarda.org/search/node/{query}"},
        {"name": "PubAg", "url": "https://pubag.nal.usda.gov/catalog/search?q={query}"}
    ]
    
    all_sources = ADDITIONAL_SOURCES + specialized_sources
    
    for source in all_sources:
        if additional_count >= PDF_COUNT_PER_DISEASE:
            break
            
        logger.info(f"Searching additional source: {source['name']} for {disease_name}")
        
        # Try multiple queries for each source
        for query in queries[:2]:  # Use first two queries for each source
            if additional_count >= PDF_COUNT_PER_DISEASE:
                break
                
            url = source['url'].format(query=quote_plus(query))
            
            try:
                headers = {'User-Agent': get_random_user_agent()}
                response = requests.get(url, headers=headers, timeout=30)
                
                if not response.ok:
                    logger.warning(f"Failed to access {source['name']} (Status: {response.status_code})")
                    continue
                    
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Look for PDF links
                pdf_links = []
                
                # General PDF link search
                for link in soup.select('a'):
                    href = link.get('href', '')
                    if not href:
                        continue
                        
                    text = link.text.lower()
                    
                    # Check if it's likely a PDF link
                    if (href.endswith('.pdf') or 
                        ('pdf' in text and ('download' in text or 'view' in text or 'full' in text)) or
                        ('pdf' in href.lower())):
                        pdf_url = urljoin(url, href)
                        pdf_links.append(pdf_url)
                
                # Also look for article pages that might contain PDFs
                article_links = []
                for link in soup.select('a'):
                    href = link.get('href', '')
                    if not href:
                        continue
                        
                    text = link.text.lower()
                    url_parts = urlparse(href)
                    
                    # Skip navigation/menu links
                    if (not url_parts.path or 
                        url_parts.path == '/' or 
                        'search' in url_parts.path.lower() or
                        'login' in url_parts.path.lower()):
                        continue
                        
                    # Check if it looks like an article link
                    if ('article' in href.lower() or 
                        'publication' in href.lower() or
                        'research' in href.lower() or
                        ('wheat' in text and disease_name.replace('_', ' ') in text)):
                        article_url = urljoin(url, href)
                        article_links.append(article_url)
                
                # Visit first few article pages to find PDFs
                for article_url in article_links[:5]:  # Limit to first 5 article links
                    if additional_count >= PDF_COUNT_PER_DISEASE:
                        break
                        
                    try:
                        # Skip if we already processed this URL
                        if article_url in downloaded_urls:
                            continue
                            
                        logger.info(f"Checking article page: {article_url}")
                        article_response = requests.get(article_url, headers={'User-Agent': get_random_user_agent()}, timeout=20)
                        
                        if not article_response.ok:
                            continue
                            
                        article_soup = BeautifulSoup(article_response.text, 'html.parser')
                        
                        # Look for PDF links in the article page
                        for link in article_soup.select('a'):
                            href = link.get('href', '')
                            if not href:
                                continue
                                
                            text = link.text.lower()
                            
                            if (href.endswith('.pdf') or 
                                ('pdf' in text and ('download' in text or 'view' in text or 'full' in text)) or
                                ('pdf' in href.lower())):
                                pdf_url = urljoin(article_url, href)
                                if pdf_url not in pdf_links:
                                    pdf_links.append(pdf_url)
                        
                        random_delay()
                    except Exception as e:
                        logger.error(f"Error checking article page {article_url}: {e}")
                        continue
                
                # Process found links
                for i, pdf_url in enumerate(pdf_links):
                    if additional_count >= PDF_COUNT_PER_DISEASE:
                        break
                        
                    if pdf_url in downloaded_urls:
                        continue
                    
                    # Create filename
                    base_name = f"{disease_name}_additional_{source['name']}_{i+1}"
                    safe_name = re.sub(r'[^\w\-_\. ]', '_', base_name)
                    output_path = os.path.join(disease_name, f"{safe_name}.pdf")
                    
                    logger.info(f"Downloading from {source['name']}: {pdf_url}")
                    success = download_pdf(pdf_url, output_path)
                    
                    if success:
                        additional_count += 1
                        downloaded_urls.add(pdf_url)
                        logger.info(f"Progress: {additional_count}/{PDF_COUNT_PER_DISEASE} PDFs for {disease_name}")
                    
                    random_delay()
                    
            except Exception as e:
                logger.error(f"Error searching additional source {source['name']}: {e}")
                continue
            
            random_delay()
    
    return additional_count

def scrape_google_scholar(disease):
    """
    Scrape Google Scholar for papers on a specific wheat disease
    Downloads PDFs and returns the count of successfully downloaded PDFs
    """
    disease_name = disease["name"]
    queries = disease["queries"]
    pdf_count = 0
    downloaded_urls = set()  # Track already downloaded URLs to avoid duplicates
    
    logger.info(f"Starting search for disease: {disease_name}")
    
    # Try each query until we reach the target PDF count
    for query_idx, query in enumerate(queries):
        if pdf_count >= PDF_COUNT_PER_DISEASE:
            break
            
        logger.info(f"Using query {query_idx+1}/{len(queries)}: {query}")
        
        page = 0
        has_next_page = True
        query_attempts = 0
        max_pages_per_query = 10  # Limit pages per query to avoid excessive scraping
        
        while pdf_count < PDF_COUNT_PER_DISEASE and has_next_page and page < max_pages_per_query:
            # Get search results
            results, has_next_page = scrape_google_scholar_page(query, page)
            
            if not results:
                query_attempts += 1
                logger.warning(f"No results found for {disease_name} on page {page} (query: {query})")
                if page > 0 and not has_next_page:
                    logger.info(f"No more pages available for this query")
                    break
                if query_attempts > 2:  # If we fail multiple times, try the next query
                    logger.warning(f"Moving to next query after multiple failed attempts")
                    break
                # Try next page if this one failed
                page += 1
                random_delay()
                continue
            
            # Process each result
            for result in results:
                if pdf_count >= PDF_COUNT_PER_DISEASE:
                    break
                
                pdf_url = result['url']
                
                # Skip if we've already downloaded this URL
                if pdf_url in downloaded_urls:
                    logger.info(f"Skipping already downloaded URL: {pdf_url}")
                    continue
                    
                # Create a safe filename from metadata
                base_name = f"{disease_name}_{result['author']}_{result['year']}_{pdf_count+1}"
                # Clean the filename to make it safe for all filesystems
                safe_name = re.sub(r'[^\w\-_\. ]', '_', base_name)
                output_path = os.path.join(disease_name, f"{safe_name}.pdf")
                
                logger.info(f"Downloading {pdf_count+1}/{PDF_COUNT_PER_DISEASE}: {pdf_url}")
                success = download_pdf(pdf_url, output_path)
                
                if success:
                    pdf_count += 1
                    downloaded_urls.add(pdf_url)
                    logger.info(f"Progress: {pdf_count}/{PDF_COUNT_PER_DISEASE} PDFs for {disease_name}")
                
                # Random delay between downloads
                random_delay()
            
            # Move to next page
            page += 1
            
            # Random delay between pages
            random_delay()
    
    # Try using additional search engines if we haven't reached the target count
    if pdf_count < PDF_COUNT_PER_DISEASE:
        pdf_count = search_additional_sources(disease_name, queries, pdf_count, downloaded_urls)
    
    logger.info(f"Completed downloading {pdf_count} PDFs for {disease_name}")
    return pdf_count

def parallel_download_disease(disease, max_pdfs):
    """Function to download PDFs for a single disease, used for parallel processing"""
    global PDF_COUNT_PER_DISEASE
    PDF_COUNT_PER_DISEASE = max_pdfs
    return scrape_google_scholar(disease)

def main():
    """Main function to run the scraper"""
    global PDF_COUNT_PER_DISEASE
    
    parser = argparse.ArgumentParser(description='Google Scholar Scraper for Wheat Disease Research')
    parser.add_argument('--disease', choices=[d['name'] for d in DISEASES], help='Specific disease to scrape (default: all)')
    parser.add_argument('--start', type=int, default=0, help='Start index for diseases (useful for resuming)')
    parser.add_argument('--count', type=int, default=PDF_COUNT_PER_DISEASE, 
                      help=f'Number of PDFs to download per disease (default: {PDF_COUNT_PER_DISEASE})')
    parser.add_argument('--parallel', action='store_true', help='Enable parallel downloading for different diseases')
    parser.add_argument('--search-exact', action='store_true', help='Use exact phrase matching for search queries')
    args = parser.parse_args()
    
    # Update count if specified
    if args.count > 0:
        PDF_COUNT_PER_DISEASE = args.count
    
    # Create output directories
    create_output_dirs()
    
    # Configure which diseases to scrape
    diseases_to_scrape = [d for d in DISEASES if args.disease is None or d['name'] == args.disease]
    
    # Start from the specified index
    if args.start > 0 and args.start < len(diseases_to_scrape):
        diseases_to_scrape = diseases_to_scrape[args.start:]
    
    logger.info(f"Starting scraping for wheat diseases: {', '.join(d['name'] for d in diseases_to_scrape)}")
    
    # Check if we should use parallel processing
    if args.parallel and len(diseases_to_scrape) > 1:
        logger.info("Using parallel processing for multiple diseases")
        
        # Use ThreadPoolExecutor to download in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(diseases_to_scrape), 4)) as executor:
            # Submit all diseases for processing
            future_to_disease = {
                executor.submit(parallel_download_disease, disease, PDF_COUNT_PER_DISEASE): disease['name'] 
                for disease in diseases_to_scrape
            }
            
            # Collect results as they complete
            total_pdfs = 0
            for future in concurrent.futures.as_completed(future_to_disease):
                disease_name = future_to_disease[future]
                try:
                    pdf_count = future.result()
                    total_pdfs += pdf_count
                    logger.info(f"Completed {disease_name} with {pdf_count} PDFs")
                except Exception as e:
                    logger.error(f"Error processing {disease_name}: {e}")
    else:
        # Process sequentially
        total_pdfs = 0
        for disease in diseases_to_scrape:
            pdf_count = scrape_google_scholar(disease)
            total_pdfs += pdf_count
    
    logger.info(f"Scraping completed. Total PDFs downloaded: {total_pdfs}")

if __name__ == "__main__":
    main()