import json
import time
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import traceback

class LDSHymnCrawler:
    def __init__(self):
        self.base_url = "https://www.churchofjesuschrist.org/music/library/hymns"
        self.hymn_data = {}
        try:
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920x1080')
            chrome_options.add_argument('--ignore-certificate-errors')
            print("Installing ChromeDriver...")
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.wait = WebDriverWait(self.driver, 10)
            print("Browser initialized successfully")
        except Exception as e:
            print(f"Error initializing browser: {e}")
            if hasattr(self, 'driver'):
                self.driver.quit()
            raise

    def get_hymn_list(self):
        """Get the list of all hymns from the website."""
        try:
            print("Navigating to hymns page...")
            self.driver.get(self.base_url)
            time.sleep(5)  # Wait for JavaScript to load content
            
            # Wait for the hymn list to load
            hymn_links = self.driver.find_elements(By.CSS_SELECTOR, "a[href*='/music/library/hymns']")
            
            # Filter and process hymn links
            hymns = []
            for link in hymn_links:
                href = link.get_attribute('href')
                if href and '/music/library/hymns/' in href and not href.endswith('hymns'):
                    try:
                        hymn_number = int(href.split('/')[-1])
                        title = link.text.strip()
                        if title:
                            hymns.append((hymn_number, title, href))
                    except ValueError:
                        continue
            
            # Sort by hymn number
            hymns.sort(key=lambda x: x[0])
            print(f"Found {len(hymns)} hymns")
            return hymns
            
        except TimeoutException:
            print("Timeout while waiting for hymn list to load")
            return []
        except Exception as e:
            print(f"Error fetching hymn list: {e}")
            return []

    def extract_musical_info(self, url):
        """Extract musical information from a hymn page."""
        try:
            print(f"Extracting musical info from {url}")
            self.driver.get(url)
            time.sleep(3)  # Wait for page to load
            
            # Extract basic information
            title = self.driver.find_element(By.TAG_NAME, "h1").text.strip()
            
            # Initialize musical info dictionary
            musical_info = {
                "title": title,
                "key_signature": "",
                "time_signature": "",
                "tempo": "",
                "meter": "",
                "author": "",
                "composer": "",
                "topics": []
            }
            
            try:
                # Try to find music metadata
                metadata_items = self.driver.find_elements(By.CLASS_NAME, "meta-item")
                for item in metadata_items:
                    text = item.text.strip()
                    
                    if "Key:" in text:
                        musical_info["key_signature"] = text.split("Key:")[-1].strip()
                    elif "Time:" in text:
                        musical_info["time_signature"] = text.split("Time:")[-1].strip()
                    elif "Tempo:" in text:
                        musical_info["tempo"] = text.split("Tempo:")[-1].strip()
                    elif "Meter:" in text:
                        musical_info["meter"] = text.split("Meter:")[-1].strip()
                
                # Extract author and composer
                credits = self.driver.find_elements(By.CLASS_NAME, "credit")
                for credit in credits:
                    text = credit.text
                    if "Text:" in text:
                        musical_info["author"] = text.split("Text:")[-1].strip()
                    elif "Music:" in text:
                        musical_info["composer"] = text.split("Music:")[-1].strip()
                
                # Extract topics
                topics = self.driver.find_elements(By.CLASS_NAME, "topic")
                musical_info["topics"] = [topic.text.strip() for topic in topics if topic.text.strip()]
                
            except Exception as e:
                print(f"Error extracting metadata: {e}")
            
            return musical_info
            
        except Exception as e:
            print(f"Error processing hymn page: {e}")
            return None
                "markings": [],
                "expression": ""
            },
            "musical_elements": {
                "form": "",
                "texture": "",
                "harmony": {
                    "primary_chords": [],
                    "cadence": ""
                },
                "rhythmic_features": [],
                "melodic_character": ""
            }
        }

        try:
            # Extract musical metadata
            metadata_div = soup.find('div', class_='music-meta')
            if metadata_div:
                meta_items = metadata_div.find_all('li')
                for item in meta_items:
                    text = item.text.strip()
                    if 'Key:' in text:
                        musical_info['key_signature'] = text.replace('Key:', '').strip()
                    elif 'Time Signature:' in text:
                        musical_info['time_signature'] = text.replace('Time Signature:', '').strip()
                    elif 'Tempo:' in text:
                        musical_info['tempo'] = text.replace('Tempo:', '').strip()
                    elif 'Meter:' in text:
                        musical_info['meter'] = text.replace('Meter:', '').strip()

            # Extract expression marks and dynamics
            expression_div = soup.find('div', class_='music-expression')
            if expression_div:
                musical_info['dynamics']['expression'] = expression_div.text.strip()

            # Look for additional musical elements
            score_div = soup.find('div', class_='music-score')
            if score_div:
                # Extract dynamics markings
                dynamics_marks = score_div.find_all(class_='dynamic-mark')
                musical_info['dynamics']['markings'] = [mark.text.strip() for mark in dynamics_marks]

                # Try to determine the opening dynamic
                first_dynamic = next((mark.text.strip() for mark in dynamics_marks), '')
                if first_dynamic:
                    musical_info['dynamics']['opening'] = first_dynamic

        except Exception as e:
            print(f"Error extracting musical information: {e}")

        return musical_info

    def crawl_hymns(self, start_num=1, end_num=5):
        """Crawl hymn information for a range of hymn numbers."""
        try:
            # First get the list of all hymns
            hymns = self.get_hymn_list()
            
            for hymn_num, (title, uri) in enumerate(hymns[start_num-1:end_num], start=start_num):
                try:
                    params = {
                        'lang': 'eng',
                        'uri': uri,
                        'platform': 'web',
                        'domains': 'churchofjesuschrist.org'
                    }
                    response = requests.get(self.base_url, headers=self.headers, params=params)
                    response.raise_for_status()
                    data = response.json()
                    
                    # Extract music metadata from JSON
                    music_data = data.get('content', {}).get('music', {})
                    
                    # Get musical information
                    musical_info = {
                        "key_signature": music_data.get('keySignature', ''),
                        "time_signature": music_data.get('timeSignature', ''),
                        "tempo": music_data.get('tempo', ''),
                        "meter": music_data.get('meter', ''),
                        "musical_setting": {
                            "harmonization": "SATB",
                            "ranges": music_data.get('ranges', {})
                        },
                        "dynamics": {
                            "opening": music_data.get('openingDynamic', ''),
                            "markings": music_data.get('dynamicMarkings', []),
                            "expression": music_data.get('expression', '')
                        },
                        "musical_elements": {
                            "form": music_data.get('form', ''),
                            "texture": music_data.get('texture', ''),
                            "harmony": {
                                "primary_chords": music_data.get('primaryChords', []),
                                "cadence": music_data.get('cadence', '')
                            },
                            "rhythmic_features": music_data.get('rhythmicFeatures', []),
                            "melodic_character": music_data.get('melodicCharacter', '')
                        }
                    }
                    
                    # Store the data
                    self.hymn_data[str(hymn_num)] = {
                        "title": title,
                        **musical_info
                    }
                    
                    print(f"Successfully crawled hymn #{hymn_num}: {title}")
                    
                    # Be polite with the server
                    time.sleep(2)
                    
                except Exception as e:
                    print(f"Error crawling hymn #{hymn_num}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error in crawl_hymns: {e}")

    def save_to_json(self, filename='lds_hymns.json'):
        """Save the crawled data to a JSON file."""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.hymn_data, f, indent=4, ensure_ascii=False)
            print(f"Successfully saved data to {filename}")
        except Exception as e:
            print(f"Error saving to JSON: {e}")

def main():
    crawler = LDSHymnCrawler()
    
    # First get the list of all hymns
    print("Getting list of hymns...")
    hymn_list = crawler.get_hymn_list()
    print(f"Found {len(hymn_list)} hymns")
    
    # Crawl the first 5 hymns (you can adjust the range)
    print("\nStarting to crawl hymns...")
    crawler.crawl_hymns(1, 5)
    
    # Save the results
    crawler.save_to_json()

if __name__ == "__main__":
    main()