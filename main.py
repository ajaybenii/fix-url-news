import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
import os
import pandas as pd
import csv
import random
import time
import logging
from google import genai
from google.genai import types

# Streamlit app configuration
st.set_page_config(page_title="News Summarizer", layout="wide", initial_sidebar_state="expanded")

# Configure logging
logging.basicConfig(
    filename='scraper.log',
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize Gemini client
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './sqy-prod.json'
gemini_client = genai.Client(
    http_options=types.HttpOptions(api_version="v1beta1"),
    vertexai=True,
    project='sqy-prod',
    location='global'
)

gemini_tools = [types.Tool(google_search=types.GoogleSearch())]

# List of user-agents to rotate
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15"
]

# Optional proxy configuration
PROXIES = {}

# Extract city and locality using Gemini
def extract_city_locality(text: str) -> dict:
    try:
        prompt = f"""
        Analyze the following news text and extract the city and locality mentioned. If none are found, return 'Unknown' for both.
        Return in JSON format: {{"city": "city_name", "locality": "locality_name"}}.
        Examples:
        - Text mentions "Gurgaon, Sector 45": {{"city": "Gurgaon", "locality": "Sector 45"}}
        - Text mentions "Noida" but no locality: {{"city": "Noida", "locality": "Unknown"}}
        - No locations mentioned: {{"city": "Unknown", "locality": "Unknown"}}

        Text: {text}
        """
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash-lite-preview-06-17",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=gemini_tools,
                max_output_tokens=100,
                system_instruction="Extract city and locality from the provided news text accurately.",
                temperature=0.7,
            )
        )
        # Assuming response.text is a JSON string
        result = eval(response.text.strip())  # Use eval for simplicity; consider json.loads for safety
        return result
    except Exception as e:
        logging.error(f"Error extracting city/locality: {e}")
        return {"city": "Unknown", "locality": "Unknown"}

# Classify news type using Gemini
def classify_news_type(text: str) -> str:
    try:
        prompt = f"""
        Analyze the following news text and classify it into one of these categories: Civic, Infrastructure, Real Estate, Development, or Other.
        Return only the category name.
        Examples:
        - News about road construction: Infrastructure
        - News about housing projects: Real Estate
        - News about city governance: Civic
        - News about urban planning: Development

        Text: {text}
        """
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash-lite-preview-06-17",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=gemini_tools,
                max_output_tokens=50,
                system_instruction="Classify the news text into a single category.",
                temperature=0.7,
            )
        )
        return response.text.strip()
    except Exception as e:
        logging.error(f"Error classifying news type: {e}")
        return "Other"

# Fetch and extract text for constructionworld.in
def fetch_and_extract_text_constructionworld(url: str) -> str:
    try:
        headers = {"User-Agent": random.choice(USER_AGENTS)}
        response = requests.get(url, headers=headers, proxies=PROXIES, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        element = soup.select_one('#content > div > div.mobile-banner')
        if element is None:
            # logging.warning(f"No content found for {url}")
            return None
        text = element.get_text()
        remove_space = text.replace("\n", '')
        result_text = re.sub(r"[\([{})\]]", "", remove_space)
        return result_text.strip() if result_text.strip() else None
    except Exception as e:
        logging.error(f"Error fetching content from {url}: {e}")
        return None

# Fetch and extract text for realty.economictimes.indiatimes.com
def fetch_and_extract_text_economic_times(url: str) -> str:
    try:
        headers = {"User-Agent": random.choice(USER_AGENTS)}
        response = requests.get(url, headers=headers, proxies=PROXIES, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        text_data = soup.get_text()
        final_result = text_data.replace("\n", '')
        description = re.sub(r"[\([{})\]]", "", final_result)
        return description.strip() if description.strip() else None
    except Exception as e:
        logging.error(f"Error fetching content from {url}: {e}")
        return None

# Fetch and extract text for rprealtyplus.com
def fetch_and_extract_text_realtyplus(url: str) -> str:
    try:
        headers = {"User-Agent": random.choice(USER_AGENTS)}
        response = requests.get(url, headers=headers, proxies=PROXIES, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        element = soup.select_one('body > div.col-md-12.p-0 > div.container.mb-4.stry-mt.mob-p-0 > div > div.col-md-8.rightSidebar.mob-p-0 > div > div > div:nth-child(4) > div')
        if element is None:
            # logging.warning(f"No content found for {url}")
            return None
        text = element.get_text()
        remove_space = text.replace("\n", '')
        result_text = re.sub(r"[\([{})\]]", "", remove_space)
        return result_text.strip() if result_text.strip() else None
    except Exception as e:
        logging.error(f"Error fetching content from {url}: {e}")
        return None

# Generate summary using Gemini
def generate_summary(text: str, url: str) -> dict:
    try:
        if not text:
            return {
                "summary": "Unable to generate summary due to missing content",
                "city": "Unknown",
                "locality": "Unknown",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "news_type": "Other"
            }
        # Extract city and locality
        location_data = extract_city_locality(text)
        # Classify news type
        news_type = classify_news_type(text)
        # Generate summary
        prompt = f"""
        You are a news summarization agent. Generate a clear, concise news summary in 20–25 words using the provided text. Focus on key facts only.
        Text: {text}
        """
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash-lite-preview-06-17",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=gemini_tools,
                max_output_tokens=300,
                system_instruction="Generate a clear, concise news summary in 20–25 words using the provided text. Focus on key facts only.",
                temperature=0.7,
            )
        )
        summary = response.text.strip()
        return {
            "summary": summary,
            "city": location_data.get("city", "Unknown"),
            "locality": location_data.get("locality", "Unknown"),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "news_type": news_type
        }
    except Exception as e:
        logging.error(f"Error generating summary for {url}: {e}")
        return {
            "summary": f"Error generating summary: {str(e)}",
            "city": "Unknown",
            "locality": "Unknown",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "news_type": "Other"
        }

# Function to fetch URLs and process news with retry logic
def fetch_news():
    news_items = []
    
    # 1. Scrape URLs from constructionworld.in
    with st.spinner("Fetching URLs from Construction World..."):
        url = "https://www.constructionworld.in"
        unique_urls = set()
        headers = {"User-Agent": random.choice(USER_AGENTS)}
        try:
            session = requests.Session()
            session.headers.update(headers)
            response = session.get(url, headers=headers, proxies=PROXIES, timeout=15)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                target_divs = soup.find_all(class_=["sidebg", "col-lg-4 col-md-12 col-sm-12 col-12"])
                for div in target_divs:
                    links = div.find_all('a', href=True)
                    for link in links:
                        href = link['href']
                        if href and not href.startswith('#') and not href.startswith('javascript:'):
                            if not href.startswith('http'):
                                href = f"{url}/{href.lstrip('/')}"
                            unique_urls.add(href)
                            if len(unique_urls) >= 25:
                                break
                    if len(unique_urls) >= 25:
                        break
                
                # Save URLs to CSV
                with open('constructionworld_urls.csv', 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['URL'])
                    for url_item in unique_urls:
                        writer.writerow([url_item])
                
                valid_urls = 0
                for news_url in unique_urls:
                    if valid_urls >= 7:
                        break
                    text = fetch_and_extract_text_constructionworld(news_url)
                    if text:
                        result = generate_summary(text, news_url)
                        news_items.append({
                            "news_url": news_url,
                            "summary": result["summary"],
                            "city": result["city"],
                            "locality": result["locality"],
                            "date": result["date"],
                            "source": "Construction World",
                            "news_type": result["news_type"]
                        })
                        valid_urls += 1
        except Exception as e:
            st.error(f"Failed to scrape Construction World: {e}")
            logging.error(f"Error scraping {url}: {e}")

    # 2. Scrape URLs from realty.economictimes.indiatimes.com RSS
    with st.spinner("Fetching URLs from Economic Times Realty..."):
        url = "https://realty.economictimes.indiatimes.com/rss/recentstories"
        try:
            headers = {"User-Agent": random.choice(USER_AGENTS)}
            response = requests.get(url, headers=headers, proxies=PROXIES, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'xml')
            guid_elements = soup.find_all("guid")
            news_urls = [guid.text for guid in guid_elements]
            
            # Save URLs to CSV
            with open('economic_times_urls.csv', 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['URL'])
                for url_item in news_urls:
                    writer.writerow([url_item])
            
            valid_urls = 0
            for news_url in news_urls:
                if valid_urls >= 7:
                    break
                text = fetch_and_extract_text_economic_times(news_url)
                if text:
                    result = generate_summary(text, news_url)
                    news_items.append({
                        "news_url": news_url,
                        "summary": result["summary"],
                        "city": result["city"],
                        "locality": result["locality"],
                        "date": result["date"],
                        "source": "Economic Times Realty",
                        "news_type": result["news_type"]
                    })
                    valid_urls += 1
        except Exception as e:
            st.error(f"Failed to scrape Economic Times Realty: {e}")
            logging.error(f"Error scraping {url}: {e}")

    # 3. Scrape URLs from rprealtyplus.com with retry logic
    with st.spinner("Fetching URLs from Realty Plus..."):
        url = "https://www.rprealtyplus.com/news-views.html"
        news_urls = []
        headers = {"User-Agent": random.choice(USER_AGENTS)}
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                session = requests.Session()
                session.headers.update(headers)
                response = session.get(url, headers=headers, proxies=PROXIES, timeout=15)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                links = soup.find_all('a', href=True)
                for link in links:
                    href = link.get('href')
                    if href and href.startswith("news-views"):
                        full_url = "https://www.rprealtyplus.com/" + href
                        if full_url not in news_urls:
                            news_urls.append(full_url)
                            if len(news_urls) >= 6:
                                break
                
                # Save URLs to CSV
                with open('realtyplus_urls.csv', 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['URL'])
                    for url_item in news_urls:
                        writer.writerow([url_item])
                
                break
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 403:
                    logging.warning(f"Attempt {attempt + 1}/{max_retries} failed with 403 Forbidden for {url}. Retrying...")
                    headers = {"User-Agent": random.choice(USER_AGENTS)}
                    time.sleep(2 ** attempt)
                    if attempt == max_retries - 1:
                        st.error(f"Failed to scrape Realty Plus after {max_retries} attempts: {e}")
                        logging.error(f"Error scraping {url} after {max_retries} attempts: {e}")
                        news_urls = []
                else:
                    st.error(f"Failed to scrape Realty Plus: {e}")
                    logging.error(f"Error scraping {url}: {e}")
                    news_urls = []
                    break
            except Exception as e:
                st.error(f"Failed to scrape Realty Plus: {e}")
                logging.error(f"Error scraping {url}: {e}")
                news_urls = []
                break
        
        valid_urls = 0
        for news_url in news_urls:
            if valid_urls >= 5:
                break
            text = fetch_and_extract_text_realtyplus(news_url)
            if text:
                result = generate_summary(text, news_url)
                news_items.append({
                    "news_url": news_url,
                    "summary": result["summary"],
                    "city": result["city"],
                    "locality": result["locality"],
                    "date": result["date"],
                    "source": "Realty Plus",
                    "news_type": result["news_type"]
                })
                valid_urls += 1

    return news_items

# Streamlit UI
st.markdown("""
    <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
        }
        .news-item {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 10px;
            background-color: #f9f9f9;
        }
    </style>
""", unsafe_allow_html=True)

st.header("News Summarizer")
st.markdown("Fetches and summarizes news from **Construction World**, **Economic Times Realty**, and **Realty Plus**. Filter by source or date.")

# Sidebar for filters
st.sidebar.header("Filters")
source_filter = st.sidebar.multiselect(
    "Select Source",
    options=["Construction World", "Economic Times Realty", "Realty Plus"],
    default=["Construction World", "Economic Times Realty", "Realty Plus"]
)
date_filter = st.sidebar.date_input("Select Date Range", [datetime.now().date(), datetime.now().date()])

# Fetch news button
if st.button("Fetch Latest News"):
    with st.spinner("Fetching and summarizing news..."):
        news_items = fetch_news()
        
        if news_items:
            # Convert to DataFrame for filtering and CSV
            df = pd.DataFrame(news_items, columns=["news_url", "summary", "city", "locality", "date", "source", "news_type"])
            
            # Apply filters
            df['date'] = pd.to_datetime(df['date']).dt.date
            date_start, date_end = date_filter
            df = df[df['source'].isin(source_filter)]
            df = df[(df['date'] >= date_start) & (df['date'] <= date_end)]
            
            if not df.empty:
                st.subheader("News Summaries")
                # Display news items in requested format
                for _, row in df.iterrows():
                    st.markdown(
                        f"""
                        <div class="news-item">
                            <strong>City:</strong> {row['city']}<br>
                            <strong>Summary:</strong> {row['summary']}<br>
                            <strong>Locality:</strong> {row['locality']}<br>
                            <strong>News Type:</strong> {row['news_type']}<br>
                            <strong>Source URL:</strong> <a href="{row['news_url']}" target="_blank">{row['news_url']}</a><br>
                            <strong>Date:</strong> {row['date']}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.warning("No news items match the selected filters.")
        else:
            st.warning("No news items fetched. Please check the log file (scraper.log) for details.")

# JSON response toggle
if st.checkbox("Show JSON Response"):
    with st.spinner("Fetching JSON response..."):
        news_items = fetch_news()
        st.json({"news": news_items})

# Download button for CSV
if 'news_items' in locals() and news_items:
    df_download = pd.DataFrame(news_items)
    csv = df_download.to_csv(index=False)
    st.download_button(
        label="Download News as CSV",
        data=csv,
        file_name="news_summaries.csv",
        mime="text/csv"
    )

# Option to view log file
if st.checkbox("Show Log File Contents"):
    try:
        with open('scraper.log', 'r') as log_file:
            st.text_area("Log File (scraper.log)", log_file.read(), height=200)
    except FileNotFoundError:
        st.info("No log file found yet.")