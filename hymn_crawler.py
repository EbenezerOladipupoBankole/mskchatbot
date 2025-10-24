import requests
from bs4 import BeautifulSoup
import json
import time
import logging
import re
from urllib.parse import urljoin

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LDSHymnCrawler:
    def __init__(self):
        self.base_url = "https://www.churchofjesuschrist.org"
        self.music_url = f"{self.base_url}/study/manual/hymns/hymns?lang=eng"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        self.hymns = []

    def fetch_page(self, url):
        """Fetch a page with exponential backoff retry"""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=self.headers)
                response.raise_for_status()
                return response.text
            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to fetch {url}: {e}")
                    raise
                time.sleep(retry_delay)
                retry_delay *= 2
        
        return None

    def parse_hymn_page(self, hymn_url):
        """Parse individual hymn page to extract details"""
        logger.info(f"Fetching hymn: {hymn_url}")
        html = self.fetch_page(hymn_url)
        if not html:
            return None

        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract hymn details
        hymn = {
            'url': hymn_url,
            'title': '',
            'number': '',
            'lyrics': [],
            'author': '',
            'composer': '',
        }

        try:
            # Get title and number from the head
            title_meta = soup.find('meta', {'name': 'title'})
            if title_meta:
                title_text = title_meta.get('content', '')
                hymn['title'] = title_text

            # Extract lyrics from verses
            verses = soup.find_all('p', class_='line')
            if verses:
                # Get text from each verse, removing the verse number span
                lyrics = []
                current_verse = []
                
                for verse in verses:
                    verse_text = verse.get_text(separator=' ', strip=True)
                    num_span = verse.find('span', class_='verse-number')
                    
                    # If we find a verse number and have content in current_verse
                    if num_span and current_verse:
                        lyrics.append(' '.join(current_verse))
                        current_verse = []

                    # Clean up the text by removing verse numbers
                    if num_span:
                        verse_text = verse_text.replace(num_span.get_text(), '').strip()
                    current_verse.append(verse_text)

                # Add the last verse
                if current_verse:
                    lyrics.append(' '.join(current_verse))
                hymn['lyrics'] = lyrics

                # Extract number from song-number paragraph
                song_num = soup.find('p', class_='song-number')
                if song_num:
                    hymn['number'] = song_num.get_text().strip()

            # Get author and composer from citation-info div
            citation_div = soup.find('div', class_='citation-info')
            if citation_div:
                for p in citation_div.find_all('p'):
                    text = p.get_text()
                    if 'Text:' in text:
                        hymn['author'] = text.replace('Text:', '').strip()
                    elif 'Music:' in text:
                        hymn['composer'] = text.replace('Music:', '').strip()

        except Exception as e:
            logger.error(f"Error parsing hymn {hymn_url}: {e}")
            return None

        return hymn

    def crawl_hymns(self):
        """Main crawler function to get all hymns"""
        logger.info("Starting hymn crawl")
        
        try:
            # Fetch main music page
            html = self.fetch_page(self.music_url)
            if not html:
                return False

            soup = BeautifulSoup(html, 'html.parser')
            
            # Find all hymn links
            hymn_links = soup.find_all('a')
            hymn_urls = []
            
            for link in hymn_links:
                href = link.get('href', '')
                # Look for links that match the hymn pattern
                if '/study/manual/hymns/' in href and href != '/study/manual/hymns?lang=eng':
                    full_url = urljoin(self.base_url, href)
                    if full_url not in hymn_urls:  # Avoid duplicates
                        hymn_urls.append(full_url)
                        logger.info(f"Found hymn link: {full_url}")

            # Process each hymn
            for url in hymn_urls:
                hymn_data = self.parse_hymn_page(url)
                if hymn_data:
                    self.hymns.append(hymn_data)
                time.sleep(1)  # Be nice to the server

            # Save to JSON file
            self.save_hymns()
            logger.info(f"Crawl completed. Found {len(self.hymns)} hymns.")
            return True

        except Exception as e:
            logger.error(f"Crawl failed: {e}")
            return False

    def save_hymns(self):
        """Save hymns to JSON file"""
        try:
            with open('hymn_book.json', 'w', encoding='utf-8') as f:
                json.dump(self.hymns, f, ensure_ascii=False, indent=2)
            logger.info("Hymns saved to hymn_book.json")
        except Exception as e:
            logger.error(f"Failed to save hymns: {e}")

if __name__ == "__main__":
    crawler = LDSHymnCrawler()
    crawler.crawl_hymns()