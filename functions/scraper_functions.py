from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
#from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


import socket
import time
from curl_cffi import requests as cureq
from bs4 import BeautifulSoup
import json
import pandas as pd
import re
import requests
from requests import Session

import httpx
from httpx import AsyncClient
import asyncio
from aiodecorators import Semaphore
"""
This class `Scraper` contains:

Attributes: 
- `base_url: str` containing the base url of the API (https://country-leaders.onrender.com)
- `country_endpoint: str` → `/countries` endpoint to get the list of supported countries
- `leaders_endpoint: str` → `/leaders` endpoint to get the list of leaders for a specific country
- `cookies_endpoint: str` → `/cookie` endpoint to get a valid cookie to query the API
- `leaders_data: dict` is a dictionary where you store the data you retrieve before saving it into the JSON file
- `cookie: object` is the cookie object used for the API calls

Methods:
- `overviewpage_counter(url :str, headers :dict) -> int` returns the number of overview pages
- `collect_data(pages:int, headers:dict) -> list` returns the collected data from the specified number of pages and saves it in a list.
- `extract_info_1(data_collection :list) -> list` This function collects data from the data_collection list. Per property a dictionary is created and populated with the key-value pairs.
- `house_url_scraper(all_properties :list, headers :dict) -> list` function scrapes the individual houses, based on the value of the key 'id' and filtered for houses on the key 'type'.
- `dataframe(all_properties :list, file :str) -> df, csv` stores the data structure into a pandas dataframe and save to .csv file
"""
"""
def __init__(self, root_url :str):
    self.root_url = root_url # containing the base url of the API (https://country-leaders.onrender.com)
    self.overview_endpoint = self.root_url + "/countries" # /countries endpoint to get the list of supported countries
    self.leaders_endpoint = self.root_url + "/leaders"   # /leaders endpoint to get the list of leaders for a specific country
    self.leader_endpoint = self.root_url + "/leader"   # /leaders endpoint to get the leader id
    self.cookies_endpoint = root_url + "/cookie" # /cookie endpoint to get a valid cookie to query the API
    self.leaders_data: dict # is a dictionary where you store the data you retrieve before saving it into the JSON file
    self.cookie_jar: object # is the cookie object used for the API calls
"""
def knock_knock(root_url):
    '''
    DOCSTRING
    '''
        
    service = Service(executable_path=r"C:\Users\tom_v\becode_projects\Immo-Eliza\.venv_immo_eliza\Scripts\chromedriver.exe")
    driver = webdriver.Chrome(service=service)

    try:
        driver.get(root_url)

        # Wait for the shadow host element to be present
        shadow_host = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "#usercentrics-root"))
        )

        # Access the shadow root
        shadow_root = driver.execute_script('return arguments[0].shadowRoot', shadow_host)

        # Wait for the 'uc-accept-all-button' inside the shadow DOM to be present
        accept_button = WebDriverWait(driver, 10).until(
            lambda d: shadow_root.find_element(By.CSS_SELECTOR, "button[data-testid='uc-accept-all-button']")
        )

        # Click the button
        accept_button.click()

        # Optionally wait for the button to be clicked and further actions
        WebDriverWait(driver, 10).until(EC.staleness_of(accept_button))

        # Now wait for the search box submit button to be present and click it
        search_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, 'searchBoxSubmitButton'))
        )
        search_button.click()

        # Additional wait if necessary (optional)
        time.sleep(5)

        # Get all properties from the pages and into a list
        ###### Still need to iterate over all the 300? pages
    finally:
        driver.quit()

    return

## Getting the number of pages
def overviewpage_counter(url :str, headers :dict):
    """
    This function calculates the number of overview pages

    """ 

    url = f"https://www.immoweb.be/nl/search-results/huis/te-koop?countries=BE&page=1&orderBy=relevance"

    resp = cureq.get(url, headers=headers, impersonate="chrome")

    #print(resp.status_code)

    data = resp.json()
    #print(data['range'])
    #print("Data keys:", data.keys())

    total_houses_1_page = int(data['range'].split('-')[1])

    total_number_of_houses= data['totalItems']

    number_of_pages = total_number_of_houses//total_houses_1_page

        #time.sleep(1)
    return number_of_pages

def collect_data(pages:int, headers:dict):
    """
    This function collects data from the specified number of pages and saves it in a list.

    Args:
    pages (int): Number of pages to collect data from.
    headers (dict): HTTP headers to include in the requests.
    """
    data_collection = []

    print("Start scraping of overview pages")

    # Loop through the pages and collect data
    for page in range(1, pages + 1):
        url = f"https://www.immoweb.be/nl/search-results/huis/te-koop?countries=BE&page={page}&orderBy=relevance"
        
        # Send the request
        resp = cureq.get(url, headers=headers, impersonate="chrome")
        
        # Append the JSON response to the list
        data_collection.append(resp.json())

    return data_collection

def url_finder(number_of_pages):

    headers = {"User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"}
    
    url_list = [] #initiate a list to capture all the urls on all overview pages

    for i in range(number_of_pages):
        page_url = f"https://www.immoweb.be/en/search/house/for-sale?countries=BE&page={i}&orderBy=relevance"
        r = requests.get(page_url, headers=headers)
        
        soup = BeautifulSoup(r.content, "html.parser")

        #elements = soup.select('a.card__title-link')
        elements = soup.find_all('a', class_='card__title-link')
                                 
        # Extract the "href" attribute from each element
        for element in elements:
            url = element.get('href')
            #print(url)
            url_list.append(url)
            
    return url_list

@Semaphore(30)
async def url_finder_async(page_url :str, session1 :AsyncClient):

    headers = {"User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"}
    
    url_list = [] #initiate a list to capture all the urls on all overview pages
    
    try:
        r = await session1.get(page_url, headers=headers, timeout=50)
    
        soup = BeautifulSoup(r.content, "html.parser")

        #elements = soup.select('a.card__title-link')
        elements = soup.find_all('a', class_='card__title-link')
                                
        # Extract the "href" attribute from each element
        for element in elements:
            url = element.get('href')
            print(url)
            url_list.append(url)

    except httpx.ConnectError as e:
        print(f"Connection error: {e} to {page_url}")
        # Handle the error, e.g., retry or return a default value
        return None  # Or handle the error differently
    
    return url_list

async def url_finder_async_main(number_of_pages :int):
    
    async with httpx.AsyncClient(timeout=60) as session1:
        tasks = []
        
        for i in range(1, number_of_pages):
            page_url = f"https://www.immoweb.be/en/search/house/for-sale?countries=BE&page={i}&orderBy=relevance"

            tasks.append(asyncio.create_task(url_finder_async(page_url, session1)))

        url_list = await asyncio.gather(*tasks)

    return url_list

def is_valid_url(url):

    try:
        response = httpx.head(url, timeout=20)  # Adjust timeout as needed
        if response.status_code == 200:
            return True
        else:
            return False
    except httpx.RequestError:
        return False

def extract_info_1(data_collection :list):
    """
    This function collects data from the data_collection list. Per property a dictionary is created and populated with the key-value pairs.

    Args:
    data_collection (list): the list with collected data generated by the function  collect_data(pages:int, headers:dict).
    """

    all_properties = []
    for page in data_collection:
        try: results_list = page['results']
        except Exception as e:
            print("Error in scraping the overview page: page>results", e)
            continue

        for i, property in enumerate(results_list):
            property = {}
            property['id'] = results_list[i]['id']
            property['type'] = results_list[i]['property']['type']

            property['subtype'] = results_list[i]['property']['subtype']
            property['country'] = results_list[i]['property']['location']['country']
            property['region'] = results_list[i]['property']['location']['region']
            property['street'] = results_list['property']['location']['street']
            property['number'] = results_list['property']['location']['number']
            property['locality_name'] = results_list[i]['property']['location']['locality']
            property['locality_code'] = results_list[i]['property']['location']['postalCode']
            
            try: property['bedroom_count'] = results_list[i]['property']['bedroomCount']
            except TypeError: property['room_count'] = "None"
            
            property['net_habitable_surface'] = results_list[i]['property']['netHabitableSurface']
            
            try: property['land_surface'] = results_list[i]['property']['landSurface']
            except TypeError: property['land_surface'] = "None"
            
            try: property['room_count'] = results_list[i]['property']['roomCount']
            except TypeError: property['room_count'] = "None"

            property['transaction_type'] = results_list[i]['transaction']['type']
            property['sale_annuity'] = results_list[i]['transaction']['sale']['lifeAnnuity']
            property['price'] = results_list[i]['transaction']['sale']['price']
            property['old_price'] = results_list[i]['transaction']['sale']['oldPrice']

            """ The other info is scraped in a second scrape, namely from the individual house ulr's --> see below """

            all_properties.append(property)
        
    print("All proporties extracted from data_collection:", len(all_properties))
    #time.sleep(2)
    return all_properties

def house_filter(all_properties :list):
    all_houses = []
    for property in all_properties:
        if 'type' in property and property['type'] == "HOUSE": # check if key property_type exists in the dictionary of the house AND that the value is 'house'
            property_url = f"https://www.immoweb.be/en/classified/{property['type']}/for-sale/{property['locality_name']}/{property['locality_code']}/{property['id']}"
            property['url'] = property_url
            all_houses.append(property)
    return all_houses

def house_filter_2(url_list :list):
    
    url_list_houses = []
    print("Started filtering houses from the list.")

    for url in url_list:
        match = re.search(r"https://www.immoweb.be/en/classified/house/for-sale/", url)
        if match:
            #print("URL of a house?:", url)
            url_list_houses.append(url)
        else:
            continue
            
    return url_list_houses

## Function to get the data of each property (filtered for houses)
def house_url_scraper(all_properties :list):
    """
    This function scrapes the indivual houses, based on the value of the key 'id' and filtered for houses on the key 'type'.
    From the house page, the javascript section is extracted. From this list of dictionaries, the data fields are extracted (including exception handling).

    Args:
    all_properties (list): list of all_properties from the overviewpage scraper (collect_data function). The list contains properties as dictionaries, prepopulated with key-value pairs from the first scrape

    """
    headers = {"User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"}
    
    house_counter = 0 # counter to display the number of houses found
    for property in all_properties:
        #time.sleep(0.76)
        
        if 'type' in property and property['type'] == "HOUSE": # check if key property_type exists in the dictionary of the house AND that the value is 'house'
            
            property_url = f"https://www.immoweb.be/en/classified/{property['type']}/for-sale/{property['locality_name']}/{property['locality_code']}/{property['id']}"
            #print("Starting scrape of URL:", property_url)

            try:
                r = requests.get(property_url, headers=headers)
                soup = BeautifulSoup(r.content, "html.parser")
                house_counter += 1

            except Exception as e:
                # Handle the exception
                print(f"Error processing {property}: {e}")
                continue
            
            # Error capture stuk van Levin hier eventueel nog zetten.
            error_tags = soup.find_all(id=re.compile(r'error', re.IGNORECASE))

            # Print the error results
            for tag in error_tags:
                print(tag)

            # Levin's approach is via javascript container main script , then to JSON
            script_tag = soup.select_one('div.classified script[type="text/javascript"]')

            if script_tag:
                # Extract the JavaScript object
                js_content = script_tag.string
                
                # Find the start and end of the JSON object
                start = js_content.find('{')
                end = js_content.rfind('}') + 1
                
                # Extract and parse the JSON data
                json_data = json.loads(js_content[start:end])
                        
                property['furnished'] = json_data['transaction']['sale']['isFurnished']

                try:
                    property['building'] = json_data['property']['building']
                    property['building_condition'] = json_data['property']['building']['condition']
                    property['facade_count'] = json_data['property']['building']['facadeCount']
                except TypeError as e:
                    #print(f"NoneTypeError gespot bij building: {e}")
                    property['building'] = "None"

                try: property['pool'] = json_data['property']['hasSwimmingPool']
                except TypeError: property['pool'] = "None"

                try:
                    property['kitchen'] = json_data['property']['kitchen']
                    property['kitchen_area'] = json_data['property']['kitchen']['surface']
                    property['kitchen_type'] = json_data['property']['kitchen']['type']
                except TypeError as e:
                    #print(f"NoneTypeError gespot bij kitchen: {e}")
                    property['kitchen'] = "None"

                try:
                    property['living_room'] = json_data['property']['livingRoom']
                    property['living_room_area'] = json_data['property']['livingRoom']['surface']
                    property['has_living_room'] = json_data['property']['hasLivingRoom']
                except TypeError as e:
                    #print(f"NoneTypeError gespot bij living room: {e}")
                    property['living_room'] = "None"

                try: property['garden'] = json_data['property']['hasGarden']
                except TypeError: property['garden'] = "None"

                try: property['garden_area'] = json_data['property']['gardenSurface']
                except TypeError: property['garden_area'] = "None"

                try: property['terrace'] = json_data['property']['hasTerrace']
                except TypeError: property['terrace'] = "None"

                try: property['terrace_area'] = json_data['property']['terraceSurface']
                except TypeError: property['terrace_area'] = "None"

                try: property['fireplace'] = json_data['property']['fireplaceExists']
                except TypeError: property['fireplace'] = "None"
                
                #property['fireplace_count'] = json_data['property']['fireplaceCount']
                        
            else:
                print("Script tag not found within the classified div.")

    return all_properties


@Semaphore(10)
async def house_url_scraper_async(url :str, session :AsyncClient, max_retries=3, base_delay=1): # 

    headers = {"User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"}
    """
    headers = {
            'Cookie': 'search_locality_code=eyJpdiI6IjU1aXM0VklyeUNRWGJ6WkR6SDZOQmc9PSIsInZhbHVlIjoiV0JDaHNkeUEzd2NaQWZEUFR5MG8xTW4wdFdOcnUxbTk2KzFYN29HeFZTVVBhM3pvUTJYTjUxZ3VZU2dmQ2lGZyIsIm1hYyI6ImY0Mjc2ZTQzYjc0OGZlNzg5ODhmMTVkNzUwOTQ5YmEyOTVjMWNhZGE5NDY3OGM3NTlmNmY0ODk0NDllZWY2ZmUifQ%3D%3D; search_property_type=eyJpdiI6IkZNb0FZQktZbHBPaENRa2svbi9LZVE9PSIsInZhbHVlIjoiTW1EbElUamRlWithZ2IwdG9WMVo0REpmcVRRY2NUNWUrci9qZ3RjS1VJNGFxT2FBWS9pZ3dLN3FtZU1GeHlGZCIsIm1hYyI6IjdkZDQyNDQ5MjEyNjYxN2I5NWY5YTc3MmM2MGJkMTg2MjFjZTdjMTdjZTY3YTE2NDk5NDE5N2RhNjIyYmVjYjcifQ%3D%3D; search_transaction_type=eyJpdiI6IjhoRXVGN2pyQW9lNGV3bXB6RnlZVUE9PSIsInZhbHVlIjoiS3FSYXA4bnEwMHROT0VBUXNFdk9xVEd2NDYxVE9rNEJRVThRQVNvejZIaVpSc2tZTkorVHZWcDFFcEU5d2w4bjgyeGhXd0Y3OEd6T09xQlFtWC9NSHc9PSIsIm1hYyI6IjE3NjZiODc0ZmVlY2VkODM2YzIwMWZhYmY5Njc5ZTAxMDc1NDc2ZmRiOWM1NTM1ZmI3ZDAzMWQ3NDdkZWFkNDMifQ%3D%3D; search_locality=eyJpdiI6Ii9ZZkNFSzYxWFM5OGgvOGNGemdaTnc9PSIsInZhbHVlIjoicHlNWVZXcVZjRllhMU9MYmJFV0JDUnlUdUxZOEx2cVQ2d3VxV0xRUjVlem5JVkV1YjdwR3lpaGxEaDB5aWFrTyIsIm1hYyI6IjVhMGUwYjljMTliNGZjMmNmYzc3NzhiMGMwYzU3NDZhNjAxZDRhMTkyNTEzZjgzZjA3NWYxOGRlM2ZlOTMxZjAifQ%3D%3D; search_province=eyJpdiI6IkI5MnlnbEdaS0k4MU9tWGt3Mk8vQkE9PSIsInZhbHVlIjoiWHdiWmVWOG84Vlkrc0Vudnd6TG5MNGNBaXRpVFFlMmRZem5IVHZDRVlMMXpubWJkZVllc2hUdnpReG85OXUxa3dqT3Y0TnA5Tk91V2FINVJtc01sUVE9PSIsIm1hYyI6IjU0YmFhNjM2OGFmZDM3NWI0MDBlMTE3YTY5YWVhMDNkODU3N2U4YTAzMDAwM2FjMTczZWJhN2NmNGM2OTc1NzIifQ%3D%3D; XSRF-TOKEN=eyJpdiI6ImpaMkZ3ZzNnU1BnNWUveEtWVkNWQlE9PSIsInZhbHVlIjoiUWs4OXlzRzl2Y05qY0wxK2g1Q0xXNEx6YWRQV01xZnkvS1ExcnNXL29ydEF6T2ZRN0Q2R2Y3Y2RXck1ieTF2SGthVlg5dC8zMkxqcENSbU9RTU81SlBQT285SEdObWJadkNFNjYvMHZEK2FkaWl6TUxjdzUra2x4UUFJaCtWd24iLCJtYWMiOiI4MmJhOTRhOWViZjBmNjk0MGQ2ZjE1MjQ4NGM3MTdiYjEwYWQxNjlmYjE5YzExNWQ1ZjQ0OTY0ODBkNDg2NjFhIn0%3D; immoweb_session=eyJpdiI6ImJuWnN1bG9GVnlzMFlrVGdLTFZoekE9PSIsInZhbHVlIjoidW12R1ZIMndSaHpCQlpIbzR4R0Q5R3AzQm9hWlRSM3lxQTdoQjBLRjdYWTg2MCtMd0FZSWQ2emdKaGNMTzYxQ3pmOTVkMHl2a2s2SmtVejVtY1owU3NXclNMUmphVERRNTk1TjAvdWdvMHJqZlI5NG42aTZiSXl2ckpUWnFTdFIiLCJtYWMiOiIyZmI4M2Q2MzE2MmQzOTgwOGM2MmRmYmRlZjdmNGJmZDA5NTY1NWRhN2Y3NDNhNTRmOTk3OTlmYjU0NDJlNDVmIn0%3D',
            'accept-language': 'nl-NL,nl;q=0.9,en-US;q=0.8,en;q=0.7',
            'client': 'WEB',
            'cookie': 'search_locality_code=eyJpdiI6IlRUNlFnSitoZC9tdzViLzZybTNCL3c9PSIsInZhbHVlIjoiK1owWkUwZXhIdFRFV0dkcWdZck5lc2ZCWnFzYU51SzhvbzBaYTlLNW5ocGd5a1BseU84TlZzTDZxa1J1b08ySCIsIm1hYyI6ImU2Y2VhNTgxOTcyZDVmZjE1YzNjZWE2NjVlOTZjNjc5ZjI5MDg1MGY3MmY5OTA5OTQyYmExODJhMTMxNzk0ZTUifQ%3D%3D; search_property_type=eyJpdiI6ImdPUlZIbWNZT0cya3NyNWRJWU5ycGc9PSIsInZhbHVlIjoib3lpQ0hVdUYwZWVuWktreWRqRVlsb2lYaTZqRXI2ck9OaDdYdHF6bmlieU84RnlueFd0WkF4Q3NEbS94ZXRWZiIsIm1hYyI6ImQwY2JiNGNlNjJmOTE4ZDA0YTFhMDE4ODViZDVkZmM3MGY0ZjUyMzM5OTY2NmNhMmQ4NDkwNTgwOTkwZGIzNTkifQ%3D%3D; search_transaction_type=eyJpdiI6IjhFbzl6WmdQK0s2ekdjTVY2WktYSkE9PSIsInZhbHVlIjoiNEdDZjM4dXFwdHU2dEJlNGRMVk5GallhYUZzQ0YrdXgzRzNsa1hyY0xvdnZQa1ppK2x2MW5HOVVuNHRtb0krV2xPZStaTmJBcHhKa2xYbXIyR2c2UVE9PSIsIm1hYyI6IjhkYTU1MTg1MDU2OTk5NTUyZTQxMGNhNDFhNmQ0MmY2NTEyYzM1NmM1NjYyZDg5MWJkYzVkOWQ1YTdlYjFkZmMifQ%3D%3D; search_locality=eyJpdiI6Ik5EUXZ2YkxQRnFzVVdJalVVeUJKOWc9PSIsInZhbHVlIjoiSzcrNFd3Nm9hUTMweG0zVFpubjVTTVllUHpnRlJvK28zcHg5ZHJic2dWQndYTDZENnhCVHN0ZWJ2Y3JVL3dVUlYzbFRrV2RRT2pFN0NoV1J6T1dFZUE9PSIsIm1hYyI6Ijg3YjVhZjczYzAxZTQxODQ4NjcxM2Q3N2Q1NDg5Y2EzNjY5NGMwNmZlYTcxZDRkMGZlNzZjMjEwZGZmNzBjNjkifQ%3D%3D; search_province=eyJpdiI6Ik1mdm53TkEwOFlhOVVzZkZBbFhQRnc9PSIsInZhbHVlIjoieVoydlc4VW9PRDhZWjliV3d0S0dnSnYxcDcwZGpCMjRlN04vRkhhRUlqTWdBWUlxQi9aOGwrTWxoeDBRRzRoTlF1TG9Tb0FwdnJtVlM5Z2hLVDBySXc9PSIsIm1hYyI6ImU0M2M1MTkwZmJkNTlhOTVmZjA2NjllMzdlMzlhNGQyNTJmYzI2ZDQ3NGRhOGNlODA5YTU4OTIxM2MyNTcyN2QifQ%3D%3D; emails_with_subscriptions_set=eyJpdiI6ImJsSnlpY0wvMnJFRWVNN1lMZ0VwR1E9PSIsInZhbHVlIjoiZWFrQ28zdURZVTFnUmtyZGFUVHpVUForM2ZReFFlZXR3Y1J6VWNwa2UrTDRkdm5TcldqWFlncHZCZ2RpVG41OFJCbDhkck5BUnIrWUVyMGUxT3hwbmRkUzlUczZUeWRQUjRyd0NBMzE4bkVRZzhzTS9zaEx0NThxNDZ0N3d3TmMiLCJtYWMiOiIzMzA3MGU0NGZkYjhkZjg1ZmI5ZTllNDc1ODQ2ZGRkNDE0MGI2ZjI5MTljZWQxNGVjN2QwZGJjODY0YjJhNTc3In0%3D; oauth_token_prod=eyJhbGciOiJSUzUxMiIsInR5cCI6IkpXVCIsImtpZCI6ImRlN2U5YmViLTYzYzMtNDNjMC04MmYzLTNlN2NlNWZkNDM5MyJ9.eyJleHAiOjE3MjY0NzQwNDQsInZlcnNpb24iOiIyMDE4LTA3LTE2IiwiaWQiOiI3NDljZGUyNy05YzI2LTQyMTMtYjJmMi1lZGQ4ZGZlMTVjMDkiLCJ1c2VybmFtZSI6ImFwcGFydGVtZW50aHVyZW5hbnR3ZXJwZW5AZ21haWwuY29tIiwic2l0ZSI6IklNTU9XRUJfQUxMIiwiZW1haWwiOiJhcHBhcnRlbWVudGh1cmVuYW50d2VycGVuQGdtYWlsLmNvbSIsImZpcnN0TmFtZSI6IlNhcyIsImxhc3ROYW1lIjoiUmlrIiwidXNlckNvcnJlbGF0aW9uSWQiOiIyYTcwMDdiY2VmY2FkMzQyZGI3OWNlZjE3OTI3Njc4OGU3YmY0NzI0ZGE1NDlhMTBhOWVjZDA0YTA3MzdkYzVlIiwibG9nZ2VkVXNlciI6eyJ1c2VybmFtZSI6ImFwcGFydGVtZW50aHVyZW5hbnR3ZXJwZW5AZ21haWwuY29tIiwieENsaWVudFNvdXJjZSI6ImltbW93ZWIuc2l0ZSIsImNvcnJlbGF0aW9uSWQiOiIyYTcwMDdiY2VmY2FkMzQyZGI3OWNlZjE3OTI3Njc4OGU3YmY0NzI0ZGE1NDlhMTBhOWVjZDA0YTA3MzdkYzVlIiwibW9kZU1hc3F1ZXJhZGUiOmZhbHNlfSwibW9kZU1hc3F1ZXJhZGUiOmZhbHNlLCJpcCI6IjU0LjE5NS4xMDIuNSIsImlzcyI6IklXQiIsInN1YiI6Ijc0OWNkZTI3LTljMjYtNDIxMy1iMmYyLWVkZDhkZmUxNWMwOSIsIm5iZiI6MTcyNjQ3MzM4NSwianRpIjoiYzA5YTE3OGMtY2RmMS00MWQ4LTg5NDYtZmRhZjc5MmQ3ZGEyIiwiYXV0aG9yaXphdGlvbnMiOnsidmVyc2lvbiI6IjIwMTgtMDctMTYiLCJzdGF0ZW1lbnRzIjpbeyJzaWQiOiJST0xFX1VTRVIiLCJlZmZlY3QiOiJBbGxvdyIsImFjdGlvbnMiOlsiKiJdLCJyZXNvdXJjZXMiOlsiNzQ5Y2RlMjctOWMyNi00MjEzLWIyZjItZWRkOGRmZTE1YzA5Il19LHsic2lkIjoiVkFMSURBVEVEIiwiZWZmZWN0IjoiQWxsb3ciLCJhY3Rpb25zIjpbIioiXSwicmVzb3VyY2VzIjpbIioiXX1dfSwibGFuZ3VhZ2UiOiJOTCIsIm9yaWdpbmFsSWQiOiIyNzY1ODkyIiwiaWF0IjoxNzI2NDczNDQ0fQ.Bg4rtmJq3n4FCEpMGnDuQ3FSJBZ8QC4Ile6EdjQIs126t0V801t29YkUyClcE-aZlmhuqqwOG2jqaPotTGQcRgxdwFhqCltQ_L8fB-UIYjK3Oo4VuIhuXxEiZ_ETvXbeCkuwTHdpzXnc3eQTfY4VUiJPhKV1Q-5Z5aZtFVyxdQQkRlKEyQ_ZBzA5w0yLt-rJG89x5PNrZZJDwF8WFksEDoJrcaIGE5M1OXefIPbrM_BWXoQGcVuglwmeHEgc32RsOn3uWCKGMaPIYY_H_mo8TXPq3f64yI-qZOiXiWg7n13T_PYkasvTzsTQG600iKDReu8Gdn0MgF_5G0FtK9TSmg; __cf_bm=Cw7CCO5_lEdSPuG0yrtXy.qztgqSW3t5wBZQdl2szAQ-1728457015-1.0.1.1-Eatd4jQgVzAmiiddf5zDyvSYLvhK9Nvu79uhbKSFAxtUhS1d1n6XK.98GmBmthhS6DJ1OP2SCsOy0k3dvKDdQw; search_locality_code=eyJpdiI6Ijh1Situcmx6MEUyLzFzaFYzMjJPVEE9PSIsInZhbHVlIjoibnhlSGxaR2ZNcWttbjNFZXpEYzRQN2ZqbDlCbUZqYmhVdEdXR0JwMFMzTmRKRTd2UFE5b1haZ0cwN3RCVUVWZyIsIm1hYyI6ImEzOGNkOTg2N2Q2MTQ5N2EyNjVhN2YxZDRkODBiOTViMDRkYjI4M2U4MTQ0ODFjYjNhM2QyMTM0OWRiMmYzYzMifQ%3D%3D; search_property_type=eyJpdiI6IkxkTkJZajFLcVZVeGE5OE92eGpSUXc9PSIsInZhbHVlIjoiUytmRS9nMXFYeEw5ZWdic0wvb3lUbDdxY21zRGRUbnBkcnY4djFGN0FnbFRTaFR4Y3RZSVBvK1ZJbHhENWZDUSIsIm1hYyI6IjM0ZGRkNjZjNWRhNDM4YjVhMjllZTg2NzM2ZWYxZmMxYTgxZWUwMmY3ZTBkNTgzNjBkM2Q4Y2Q2N2QxMjIzMjQifQ%3D%3D; search_transaction_type=eyJpdiI6Ii9Od1I3KzRERDVRWjAzQ3lXaDkzaXc9PSIsInZhbHVlIjoiRGorRWVKVkNXS2dxeW9LNFFmS2ZtK0RDd3FwRXdlM1I0bGJpSjZrNnpJOGRKYzF6aDU1QWlPS2JxdzFTeGcvek1kVjc5U3RZbURVMTR5aFh5M2t3N2c9PSIsIm1hYyI6Ijg2ZGFmOTU3YTdiMGZiYThkNTExNjQxOWY0OWYxMDQxYmJlMTg3ZjkwZjQ1N2EwZDUxNWU0YzMxODZlMjI2ZWEifQ%3D%3D; search_locality=eyJpdiI6IkpMeEkwTjBDaEI0OWsvVHZsWWt5d0E9PSIsInZhbHVlIjoiMk9tMHAzTXAwMVpjTG94cUJ2NU84QUpacTE0a1g5QVlrNTB4K0VYNWhRNVB3SENzMXBVZXI1WURWY0hiNWZCMyIsIm1hYyI6ImU1YTRiZThmNDZiOGI2M2M5ODgwNWFkZmRhZGNmMGRjYzdmMWI0ZTRjYzk0OTE5NGJhNGRkZWI5MWQ4MDhiYjcifQ%3D%3D; search_province=eyJpdiI6InZUREtWUkx6d1MxTVl5NGhLZW10eVE9PSIsInZhbHVlIjoiNTkvenAyZFlKWjFWdkdqRC8zbmtOWjJWRWV0c1Nhclo5QkFrTGd4bERpMGYxWkdNcDFuSEEvT083ZERLb1UwRW1ZZ1N3VlVRbndxTkRZNkozeHR0dlE9PSIsIm1hYyI6IjRiMzgxZTZjZTJjYWIxNTk0M2ZhZTM0Y2QyOTdlNGI0ODNmODdmZTkxNjZmMzFiYjBhYWU3ZTI1MmU0YWZiNDYifQ%3D%3D; XSRF-TOKEN=eyJpdiI6ImR1VkgyWFpzelJQNlQ1b2R1djJOQWc9PSIsInZhbHVlIjoiaWFUc1Mrb0JmSW1yTUZ2WXh6Y3ROdUZ6b0pUVnR5UDc0YmI2bTFwU2kwL0lvbmljOHN4dFdmQkpGZXBOekRBM0xSdVcwOXpKQVdOQWJIbXhwZ2FBSUZ1UU1nczZEOGZKUVJ3emo4dEg5RnVZWXdDdFMrcTIvUCsxWkVJT2pPTlYiLCJtYWMiOiIwYzczZTY2MWY5MjkzYzA0NWMyN2Y2OGNkMTI1MjQ3ODNhODY1ZWUxZDdjMTYyMDg3MmIxMzBhNzM0NDUzZTlmIn0%3D; immoweb_session=eyJpdiI6InUrcnlnWmZPaEZTVUNtNlNSWUcrNlE9PSIsInZhbHVlIjoiSWlIQVlQbGJuK0prQ0YvTXp5VmNpcFZIaVdiNWZ0VGpzUlhyZWJob3J5SDlHajl3S1lJRENldkRad2ZnaXkrR05xZ3hISVFxZUhGK1FuV2RMK2MwcmlHMWkycW95WDZET0paVHlwWTMxVjRaTkxETEtNUk51NlFrWGRuMEMyRzYiLCJtYWMiOiI0NDQ4NTQ2NjU1OTRiZjBjZTlhZGNlNTFkZmYyZDUyYmU4YjU5YzU3NzM5Y2ZhNDMxZjM4ZjExNDg3ZDhkMzU3In0%3D',
            'place': 'CLASSIFIED_SEARCH_RESULT',
            'priority': 'u=1, i',
            'referer': 'https://www.immoweb.be/nl/zoeken/huis/te-koop?countries=BE&page=1&orderBy=relevance',
            'sec-ch-ua': '"Google Chrome";v="129", "Not=A?Brand";v="8", "Chromium";v="129"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'trigger': 'USER',
            'x-xsrf-token': 'eyJpdiI6ImR1VkgyWFpzelJQNlQ1b2R1djJOQWc9PSIsInZhbHVlIjoiaWFUc1Mrb0JmSW1yTUZ2WXh6Y3ROdUZ6b0pUVnR5UDc0YmI2bTFwU2kwL0lvbmljOHN4dFdmQkpGZXBOekRBM0xSdVcwOXpKQVdOQWJIbXhwZ2FBSUZ1UU1nczZEOGZKUVJ3emo4dEg5RnVZWXdDdFMrcTIvUCsxWkVJT2pPTlYiLCJtYWMiOiIwYzczZTY2MWY5MjkzYzA0NWMyN2Y2OGNkMTI1MjQ3ODNhODY1ZWUxZDdjMTYyMDg3MmIxMzBhNzM0NDUzZTlmIn0='
            }
    """
    property = {}
    for attempt in range(max_retries):
        try:
            #print("Starting scrape of URL:", url)
            r = await session.get(url, headers=headers, timeout=50)

            soup = BeautifulSoup(r.content, "html.parser")
            
            # Error capture stuk van Levin hier eventueel nog zetten.
            error_tags = soup.find_all(id=re.compile(r'error', re.IGNORECASE))

            # Print the error results
            for tag in error_tags:
                print(tag)

        #######################HIER ZIT DUS NOG EEN PROBLEEM ##################

            # Levin's approach is via javascript container main script , then to JSON
            
            script_tag = soup.select_one('div.classified script[type="text/javascript"]')

            if script_tag:
                # Extract the JavaScript object
                js_content = script_tag.string
                
                # Find the start and end of the JSON object
                start = js_content.find('{')
                end = js_content.rfind('}') + 1
                
                # Extract and parse the JSON data
                json_data = json.loads(js_content[start:end])
                #print("JSON_data:", json_data)

                
                property['url'] = url

                property['id'] = json_data['id']
                property['type'] = json_data['property']['type']

                property['subtype'] = json_data['property']['subtype']
                property['country'] = json_data['property']['location']['country']
                property['region'] = json_data['property']['location']['region']
                property['street'] = json_data['property']['location']['street']
                property['number'] = json_data['property']['location']['number']
                property['locality_name'] = json_data['property']['location']['locality']
                property['locality_code'] = json_data['property']['location']['postalCode']
                property['locality_latitude'] = json_data['property']['location']['latitude']
                property['locality_longitude'] = json_data['property']['location']['longitude']
                
                
                try: property['bedroom_count'] = json_data['property']['bedroomCount']
                except TypeError: property['room_count'] = "None"
                
                property['net_habitable_surface'] = json_data['property']['netHabitableSurface']
                
                try:property['land_surface'] = json_data['property']['landSurface']
                except: property['land_surface'] = "None"

                try: property['room_count'] = json_data['property']['roomCount']
                except TypeError: property['room_count'] = "None"

                property['transaction_type'] = json_data['transaction']['type']
                property['sale_annuity'] = json_data['transaction']['sale']['lifeAnnuity']
                property['price'] = json_data['transaction']['sale']['price']
                property['old_price'] = json_data['transaction']['sale']['oldPrice']
                property['furnished'] = json_data['transaction']['sale']['isFurnished']
                
                try: property['epc'] = json_data['transaction']['certificates']['epcScore']
                except TypeError: property['epc'] = "None"

                try:
                    if json_data['property']['building'] != None or json_data['property']['building'] != "None":
                        property['annex_count'] = json_data['property']['building']['annexCount']
                        property['building_condition'] = json_data['property']['building']['condition']
                        property['construction_year'] = json_data['property']['building']['constructionYear']
                        property['facade_count'] = json_data['property']['building']['facadeCount']
                        property['floor_count'] = json_data['property']['building']['floorCount']            
                        property['street_facade_width'] = json_data['property']['building']['streetFacadeWidth']
                except TypeError as e:
                    #print(f"NoneTypeError gespot bij building: {e}")
                    property['building'] = "None"

                try: property['pool'] = json_data['property']['hasSwimmingPool']
                except TypeError: property['pool'] = "None"

                try:
                    if json_data['property']['kitchen'] != None or json_data['property']['kitchen'] != "None":
                        property['kitchen_area'] = json_data['property']['kitchen']['surface']
                        property['kitchen_type'] = json_data['property']['kitchen']['type']
                        property['kitchen_oven'] = json_data['property']['kitchen']['hasOven']
                        property['kitchen_microwave'] = json_data['property']['kitchen']['hasMicroWaveOven']
                        property['kitchen_dishwasher'] = json_data['property']['kitchen']['hasDishwasher']
                        property['kitchen_washing_machine'] = json_data['property']['kitchen']['hasWashingMachine']
                        property['kitchen_fridge'] = json_data['property']['kitchen']['hasFridge']
                        property['kitchen_freezer'] = json_data['property']['kitchen']['hasFreezer']
                        property['kitchen_steam_oven'] = json_data['property']['kitchen']['hasSteamOven']

                except TypeError as e:
                    #print(f"NoneTypeError gespot bij kitchen: {e}")
                    property['kitchen'] = "None"

                try:
                    if json_data['property']['livingRoom'] != None or json_data['property']['livingRoom'] != "None":
                        property['living_room_area'] = json_data['property']['livingRoom']['surface']
                        property['has_living_room'] = json_data['property']['hasLivingRoom']
                except TypeError as e:
                    #print(f"NoneTypeError gespot bij living room: {e}")
                    property['living_room'] = "None"

                try: property['garden'] = json_data['property']['hasGarden']
                except TypeError: property['garden'] = "None"

                try: property['garden_area'] = json_data['property']['gardenSurface']
                except TypeError: property['garden_area'] = "None"

                try: property['terrace'] = json_data['property']['hasTerrace']
                except TypeError: property['terrace'] = "None"

                try: property['terrace_area'] = json_data['property']['terraceSurface']
                except TypeError: property['terrace_area'] = "None"

                try: property['fireplace'] = json_data['property']['fireplaceExists']
                except TypeError: property['fireplace'] = "None"
                
                #property['fireplace_count'] = json_data['property']['fireplaceCount']     
                    
            else:
                print("Script tag not found within the classified div.")
            
            try: property = property
            except Exception as e:
                print("Error in the house scraper async module with 'property':", e)
            
            print(f"Scraped: {url}")

            return property
        
        except (httpx.ReadError, httpx.RemoteProtocolError) as e:
            print(f"Remote protocol error: {e}")
            if attempt < max_retries - 1:
                delay = base_delay * 2**attempt
                print(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
            return None
            
    return property # Or handle the error differently
        
async def process_batch(all_houses_batch, session):
    results = []

    for url in all_houses_batch:
        result = await house_url_scraper_async(url, session)
        results.append(result)

    return results

#Function main loop for asyncio
async def house_url_scraper_async_main(url_list_houses):
    
    async with httpx.AsyncClient(timeout=60) as session:
        batch_size = 50
        batches = create_batches(url_list_houses, batch_size)
        tasks = [process_batch(all_houses_batch, session) for all_houses_batch in batches]
        all_houses2 = await asyncio.gather(*tasks)

    return all_houses2

def create_batches(tasks, batch_size):
    batches = []
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i+batch_size]
        batches.append(batch)
    return batches

def list_as_list(list :list):
    list_as_list = []

    for sublist in list:
        for item in sublist:
            list_as_list.append(item)

    return list_as_list