# Versie zonder selenium om cookie en search list te klikken
# Daarna loop door alle pagina's met Beautifulsoup voor urls (en uit de urls's met regex ook de property_id, post code, location, en type proprty)
# Daarna loop door alle urls, met filter op house als type:
# Van alle pagina's json format van het javascript maken
# JSON_data extract van alle nodige info volgens opdracht, en in lijst zetten als dictionary per huis


# The selenium.webdriver module provides all the implementations of WebDriver
# Currently supported are Firefox, Chrome, IE and Remote. The `Keys` class provides keys on
# the keyboard such as RETURN, F1, ALT etc.

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
#from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

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

from functions.scraper_functions import knock_knock, overviewpage_counter, collect_data, url_finder, url_finder_async_main, extract_info_1, house_filter, house_filter_2, house_url_scraper, house_url_scraper_async_main, house_url_scraper_async, list_as_list
from utils.utils import dataframe_to_csv

headers = {
            'Cookie': 'search_postal_code=eyJpdiI6IjU1aXM0VklyeUNRWGJ6WkR6SDZOQmc9PSIsInZhbHVlIjoiV0JDaHNkeUEzd2NaQWZEUFR5MG8xTW4wdFdOcnUxbTk2KzFYN29HeFZTVVBhM3pvUTJYTjUxZ3VZU2dmQ2lGZyIsIm1hYyI6ImY0Mjc2ZTQzYjc0OGZlNzg5ODhmMTVkNzUwOTQ5YmEyOTVjMWNhZGE5NDY3OGM3NTlmNmY0ODk0NDllZWY2ZmUifQ%3D%3D; search_property_type=eyJpdiI6IkZNb0FZQktZbHBPaENRa2svbi9LZVE9PSIsInZhbHVlIjoiTW1EbElUamRlWithZ2IwdG9WMVo0REpmcVRRY2NUNWUrci9qZ3RjS1VJNGFxT2FBWS9pZ3dLN3FtZU1GeHlGZCIsIm1hYyI6IjdkZDQyNDQ5MjEyNjYxN2I5NWY5YTc3MmM2MGJkMTg2MjFjZTdjMTdjZTY3YTE2NDk5NDE5N2RhNjIyYmVjYjcifQ%3D%3D; search_transaction_type=eyJpdiI6IjhoRXVGN2pyQW9lNGV3bXB6RnlZVUE9PSIsInZhbHVlIjoiS3FSYXA4bnEwMHROT0VBUXNFdk9xVEd2NDYxVE9rNEJRVThRQVNvejZIaVpSc2tZTkorVHZWcDFFcEU5d2w4bjgyeGhXd0Y3OEd6T09xQlFtWC9NSHc9PSIsIm1hYyI6IjE3NjZiODc0ZmVlY2VkODM2YzIwMWZhYmY5Njc5ZTAxMDc1NDc2ZmRiOWM1NTM1ZmI3ZDAzMWQ3NDdkZWFkNDMifQ%3D%3D; search_locality=eyJpdiI6Ii9ZZkNFSzYxWFM5OGgvOGNGemdaTnc9PSIsInZhbHVlIjoicHlNWVZXcVZjRllhMU9MYmJFV0JDUnlUdUxZOEx2cVQ2d3VxV0xRUjVlem5JVkV1YjdwR3lpaGxEaDB5aWFrTyIsIm1hYyI6IjVhMGUwYjljMTliNGZjMmNmYzc3NzhiMGMwYzU3NDZhNjAxZDRhMTkyNTEzZjgzZjA3NWYxOGRlM2ZlOTMxZjAifQ%3D%3D; search_province=eyJpdiI6IkI5MnlnbEdaS0k4MU9tWGt3Mk8vQkE9PSIsInZhbHVlIjoiWHdiWmVWOG84Vlkrc0Vudnd6TG5MNGNBaXRpVFFlMmRZem5IVHZDRVlMMXpubWJkZVllc2hUdnpReG85OXUxa3dqT3Y0TnA5Tk91V2FINVJtc01sUVE9PSIsIm1hYyI6IjU0YmFhNjM2OGFmZDM3NWI0MDBlMTE3YTY5YWVhMDNkODU3N2U4YTAzMDAwM2FjMTczZWJhN2NmNGM2OTc1NzIifQ%3D%3D; XSRF-TOKEN=eyJpdiI6ImpaMkZ3ZzNnU1BnNWUveEtWVkNWQlE9PSIsInZhbHVlIjoiUWs4OXlzRzl2Y05qY0wxK2g1Q0xXNEx6YWRQV01xZnkvS1ExcnNXL29ydEF6T2ZRN0Q2R2Y3Y2RXck1ieTF2SGthVlg5dC8zMkxqcENSbU9RTU81SlBQT285SEdObWJadkNFNjYvMHZEK2FkaWl6TUxjdzUra2x4UUFJaCtWd24iLCJtYWMiOiI4MmJhOTRhOWViZjBmNjk0MGQ2ZjE1MjQ4NGM3MTdiYjEwYWQxNjlmYjE5YzExNWQ1ZjQ0OTY0ODBkNDg2NjFhIn0%3D; immoweb_session=eyJpdiI6ImJuWnN1bG9GVnlzMFlrVGdLTFZoekE9PSIsInZhbHVlIjoidW12R1ZIMndSaHpCQlpIbzR4R0Q5R3AzQm9hWlRSM3lxQTdoQjBLRjdYWTg2MCtMd0FZSWQ2emdKaGNMTzYxQ3pmOTVkMHl2a2s2SmtVejVtY1owU3NXclNMUmphVERRNTk1TjAvdWdvMHJqZlI5NG42aTZiSXl2ckpUWnFTdFIiLCJtYWMiOiIyZmI4M2Q2MzE2MmQzOTgwOGM2MmRmYmRlZjdmNGJmZDA5NTY1NWRhN2Y3NDNhNTRmOTk3OTlmYjU0NDJlNDVmIn0%3D',
            'accept-language': 'nl-NL,nl;q=0.9,en-US;q=0.8,en;q=0.7',
            'client': 'WEB',
            'cookie': 'search_postal_code=eyJpdiI6IlRUNlFnSitoZC9tdzViLzZybTNCL3c9PSIsInZhbHVlIjoiK1owWkUwZXhIdFRFV0dkcWdZck5lc2ZCWnFzYU51SzhvbzBaYTlLNW5ocGd5a1BseU84TlZzTDZxa1J1b08ySCIsIm1hYyI6ImU2Y2VhNTgxOTcyZDVmZjE1YzNjZWE2NjVlOTZjNjc5ZjI5MDg1MGY3MmY5OTA5OTQyYmExODJhMTMxNzk0ZTUifQ%3D%3D; search_property_type=eyJpdiI6ImdPUlZIbWNZT0cya3NyNWRJWU5ycGc9PSIsInZhbHVlIjoib3lpQ0hVdUYwZWVuWktreWRqRVlsb2lYaTZqRXI2ck9OaDdYdHF6bmlieU84RnlueFd0WkF4Q3NEbS94ZXRWZiIsIm1hYyI6ImQwY2JiNGNlNjJmOTE4ZDA0YTFhMDE4ODViZDVkZmM3MGY0ZjUyMzM5OTY2NmNhMmQ4NDkwNTgwOTkwZGIzNTkifQ%3D%3D; search_transaction_type=eyJpdiI6IjhFbzl6WmdQK0s2ekdjTVY2WktYSkE9PSIsInZhbHVlIjoiNEdDZjM4dXFwdHU2dEJlNGRMVk5GallhYUZzQ0YrdXgzRzNsa1hyY0xvdnZQa1ppK2x2MW5HOVVuNHRtb0krV2xPZStaTmJBcHhKa2xYbXIyR2c2UVE9PSIsIm1hYyI6IjhkYTU1MTg1MDU2OTk5NTUyZTQxMGNhNDFhNmQ0MmY2NTEyYzM1NmM1NjYyZDg5MWJkYzVkOWQ1YTdlYjFkZmMifQ%3D%3D; search_locality=eyJpdiI6Ik5EUXZ2YkxQRnFzVVdJalVVeUJKOWc9PSIsInZhbHVlIjoiSzcrNFd3Nm9hUTMweG0zVFpubjVTTVllUHpnRlJvK28zcHg5ZHJic2dWQndYTDZENnhCVHN0ZWJ2Y3JVL3dVUlYzbFRrV2RRT2pFN0NoV1J6T1dFZUE9PSIsIm1hYyI6Ijg3YjVhZjczYzAxZTQxODQ4NjcxM2Q3N2Q1NDg5Y2EzNjY5NGMwNmZlYTcxZDRkMGZlNzZjMjEwZGZmNzBjNjkifQ%3D%3D; search_province=eyJpdiI6Ik1mdm53TkEwOFlhOVVzZkZBbFhQRnc9PSIsInZhbHVlIjoieVoydlc4VW9PRDhZWjliV3d0S0dnSnYxcDcwZGpCMjRlN04vRkhhRUlqTWdBWUlxQi9aOGwrTWxoeDBRRzRoTlF1TG9Tb0FwdnJtVlM5Z2hLVDBySXc9PSIsIm1hYyI6ImU0M2M1MTkwZmJkNTlhOTVmZjA2NjllMzdlMzlhNGQyNTJmYzI2ZDQ3NGRhOGNlODA5YTU4OTIxM2MyNTcyN2QifQ%3D%3D; emails_with_subscriptions_set=eyJpdiI6ImJsSnlpY0wvMnJFRWVNN1lMZ0VwR1E9PSIsInZhbHVlIjoiZWFrQ28zdURZVTFnUmtyZGFUVHpVUForM2ZReFFlZXR3Y1J6VWNwa2UrTDRkdm5TcldqWFlncHZCZ2RpVG41OFJCbDhkck5BUnIrWUVyMGUxT3hwbmRkUzlUczZUeWRQUjRyd0NBMzE4bkVRZzhzTS9zaEx0NThxNDZ0N3d3TmMiLCJtYWMiOiIzMzA3MGU0NGZkYjhkZjg1ZmI5ZTllNDc1ODQ2ZGRkNDE0MGI2ZjI5MTljZWQxNGVjN2QwZGJjODY0YjJhNTc3In0%3D; oauth_token_prod=eyJhbGciOiJSUzUxMiIsInR5cCI6IkpXVCIsImtpZCI6ImRlN2U5YmViLTYzYzMtNDNjMC04MmYzLTNlN2NlNWZkNDM5MyJ9.eyJleHAiOjE3MjY0NzQwNDQsInZlcnNpb24iOiIyMDE4LTA3LTE2IiwiaWQiOiI3NDljZGUyNy05YzI2LTQyMTMtYjJmMi1lZGQ4ZGZlMTVjMDkiLCJ1c2VybmFtZSI6ImFwcGFydGVtZW50aHVyZW5hbnR3ZXJwZW5AZ21haWwuY29tIiwic2l0ZSI6IklNTU9XRUJfQUxMIiwiZW1haWwiOiJhcHBhcnRlbWVudGh1cmVuYW50d2VycGVuQGdtYWlsLmNvbSIsImZpcnN0TmFtZSI6IlNhcyIsImxhc3ROYW1lIjoiUmlrIiwidXNlckNvcnJlbGF0aW9uSWQiOiIyYTcwMDdiY2VmY2FkMzQyZGI3OWNlZjE3OTI3Njc4OGU3YmY0NzI0ZGE1NDlhMTBhOWVjZDA0YTA3MzdkYzVlIiwibG9nZ2VkVXNlciI6eyJ1c2VybmFtZSI6ImFwcGFydGVtZW50aHVyZW5hbnR3ZXJwZW5AZ21haWwuY29tIiwieENsaWVudFNvdXJjZSI6ImltbW93ZWIuc2l0ZSIsImNvcnJlbGF0aW9uSWQiOiIyYTcwMDdiY2VmY2FkMzQyZGI3OWNlZjE3OTI3Njc4OGU3YmY0NzI0ZGE1NDlhMTBhOWVjZDA0YTA3MzdkYzVlIiwibW9kZU1hc3F1ZXJhZGUiOmZhbHNlfSwibW9kZU1hc3F1ZXJhZGUiOmZhbHNlLCJpcCI6IjU0LjE5NS4xMDIuNSIsImlzcyI6IklXQiIsInN1YiI6Ijc0OWNkZTI3LTljMjYtNDIxMy1iMmYyLWVkZDhkZmUxNWMwOSIsIm5iZiI6MTcyNjQ3MzM4NSwianRpIjoiYzA5YTE3OGMtY2RmMS00MWQ4LTg5NDYtZmRhZjc5MmQ3ZGEyIiwiYXV0aG9yaXphdGlvbnMiOnsidmVyc2lvbiI6IjIwMTgtMDctMTYiLCJzdGF0ZW1lbnRzIjpbeyJzaWQiOiJST0xFX1VTRVIiLCJlZmZlY3QiOiJBbGxvdyIsImFjdGlvbnMiOlsiKiJdLCJyZXNvdXJjZXMiOlsiNzQ5Y2RlMjctOWMyNi00MjEzLWIyZjItZWRkOGRmZTE1YzA5Il19LHsic2lkIjoiVkFMSURBVEVEIiwiZWZmZWN0IjoiQWxsb3ciLCJhY3Rpb25zIjpbIioiXSwicmVzb3VyY2VzIjpbIioiXX1dfSwibGFuZ3VhZ2UiOiJOTCIsIm9yaWdpbmFsSWQiOiIyNzY1ODkyIiwiaWF0IjoxNzI2NDczNDQ0fQ.Bg4rtmJq3n4FCEpMGnDuQ3FSJBZ8QC4Ile6EdjQIs126t0V801t29YkUyClcE-aZlmhuqqwOG2jqaPotTGQcRgxdwFhqCltQ_L8fB-UIYjK3Oo4VuIhuXxEiZ_ETvXbeCkuwTHdpzXnc3eQTfY4VUiJPhKV1Q-5Z5aZtFVyxdQQkRlKEyQ_ZBzA5w0yLt-rJG89x5PNrZZJDwF8WFksEDoJrcaIGE5M1OXefIPbrM_BWXoQGcVuglwmeHEgc32RsOn3uWCKGMaPIYY_H_mo8TXPq3f64yI-qZOiXiWg7n13T_PYkasvTzsTQG600iKDReu8Gdn0MgF_5G0FtK9TSmg; __cf_bm=Cw7CCO5_lEdSPuG0yrtXy.qztgqSW3t5wBZQdl2szAQ-1728457015-1.0.1.1-Eatd4jQgVzAmiiddf5zDyvSYLvhK9Nvu79uhbKSFAxtUhS1d1n6XK.98GmBmthhS6DJ1OP2SCsOy0k3dvKDdQw; search_postal_code=eyJpdiI6Ijh1Situcmx6MEUyLzFzaFYzMjJPVEE9PSIsInZhbHVlIjoibnhlSGxaR2ZNcWttbjNFZXpEYzRQN2ZqbDlCbUZqYmhVdEdXR0JwMFMzTmRKRTd2UFE5b1haZ0cwN3RCVUVWZyIsIm1hYyI6ImEzOGNkOTg2N2Q2MTQ5N2EyNjVhN2YxZDRkODBiOTViMDRkYjI4M2U4MTQ0ODFjYjNhM2QyMTM0OWRiMmYzYzMifQ%3D%3D; search_property_type=eyJpdiI6IkxkTkJZajFLcVZVeGE5OE92eGpSUXc9PSIsInZhbHVlIjoiUytmRS9nMXFYeEw5ZWdic0wvb3lUbDdxY21zRGRUbnBkcnY4djFGN0FnbFRTaFR4Y3RZSVBvK1ZJbHhENWZDUSIsIm1hYyI6IjM0ZGRkNjZjNWRhNDM4YjVhMjllZTg2NzM2ZWYxZmMxYTgxZWUwMmY3ZTBkNTgzNjBkM2Q4Y2Q2N2QxMjIzMjQifQ%3D%3D; search_transaction_type=eyJpdiI6Ii9Od1I3KzRERDVRWjAzQ3lXaDkzaXc9PSIsInZhbHVlIjoiRGorRWVKVkNXS2dxeW9LNFFmS2ZtK0RDd3FwRXdlM1I0bGJpSjZrNnpJOGRKYzF6aDU1QWlPS2JxdzFTeGcvek1kVjc5U3RZbURVMTR5aFh5M2t3N2c9PSIsIm1hYyI6Ijg2ZGFmOTU3YTdiMGZiYThkNTExNjQxOWY0OWYxMDQxYmJlMTg3ZjkwZjQ1N2EwZDUxNWU0YzMxODZlMjI2ZWEifQ%3D%3D; search_locality=eyJpdiI6IkpMeEkwTjBDaEI0OWsvVHZsWWt5d0E9PSIsInZhbHVlIjoiMk9tMHAzTXAwMVpjTG94cUJ2NU84QUpacTE0a1g5QVlrNTB4K0VYNWhRNVB3SENzMXBVZXI1WURWY0hiNWZCMyIsIm1hYyI6ImU1YTRiZThmNDZiOGI2M2M5ODgwNWFkZmRhZGNmMGRjYzdmMWI0ZTRjYzk0OTE5NGJhNGRkZWI5MWQ4MDhiYjcifQ%3D%3D; search_province=eyJpdiI6InZUREtWUkx6d1MxTVl5NGhLZW10eVE9PSIsInZhbHVlIjoiNTkvenAyZFlKWjFWdkdqRC8zbmtOWjJWRWV0c1Nhclo5QkFrTGd4bERpMGYxWkdNcDFuSEEvT083ZERLb1UwRW1ZZ1N3VlVRbndxTkRZNkozeHR0dlE9PSIsIm1hYyI6IjRiMzgxZTZjZTJjYWIxNTk0M2ZhZTM0Y2QyOTdlNGI0ODNmODdmZTkxNjZmMzFiYjBhYWU3ZTI1MmU0YWZiNDYifQ%3D%3D; XSRF-TOKEN=eyJpdiI6ImR1VkgyWFpzelJQNlQ1b2R1djJOQWc9PSIsInZhbHVlIjoiaWFUc1Mrb0JmSW1yTUZ2WXh6Y3ROdUZ6b0pUVnR5UDc0YmI2bTFwU2kwL0lvbmljOHN4dFdmQkpGZXBOekRBM0xSdVcwOXpKQVdOQWJIbXhwZ2FBSUZ1UU1nczZEOGZKUVJ3emo4dEg5RnVZWXdDdFMrcTIvUCsxWkVJT2pPTlYiLCJtYWMiOiIwYzczZTY2MWY5MjkzYzA0NWMyN2Y2OGNkMTI1MjQ3ODNhODY1ZWUxZDdjMTYyMDg3MmIxMzBhNzM0NDUzZTlmIn0%3D; immoweb_session=eyJpdiI6InUrcnlnWmZPaEZTVUNtNlNSWUcrNlE9PSIsInZhbHVlIjoiSWlIQVlQbGJuK0prQ0YvTXp5VmNpcFZIaVdiNWZ0VGpzUlhyZWJob3J5SDlHajl3S1lJRENldkRad2ZnaXkrR05xZ3hISVFxZUhGK1FuV2RMK2MwcmlHMWkycW95WDZET0paVHlwWTMxVjRaTkxETEtNUk51NlFrWGRuMEMyRzYiLCJtYWMiOiI0NDQ4NTQ2NjU1OTRiZjBjZTlhZGNlNTFkZmYyZDUyYmU4YjU5YzU3NzM5Y2ZhNDMxZjM4ZjExNDg3ZDhkMzU3In0%3D',
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

root_url = "https://www.immoweb.be/nl"

start_time = time.perf_counter()

#running the function to deal with cookies and getting to the overview pages (selenium)
#knock_knock(root_url)

lap_time_1 = time.perf_counter()

#running the overviewpage_counter function
number_of_pages = overviewpage_counter("https://www.immoweb.be/nl/search-results/huis/te-koop?countries=BE&page=1&orderBy=relevance", headers)

lap_time_2 = time.perf_counter()

## Looping through the pages and start collecting the data
#running the collect function

#data_collection = collect_data(number_of_pages, headers)
#url_list = url_finder(2)

with Session() as session:
    url_list = asyncio.run(url_finder_async_main(number_of_pages)) #asyncio with batches

lap_time_3 = time.perf_counter()

#run the function extract_info_1 to get the key-value pairs out of the data_collection list and return an all_properties list
#all_properties = extract_info_1(data_collection)

#run the function to create, printing and save to csv the (first = complete) dataframe
#dataframe(all_properties, "./data/immoweb_scrape_1.csv")

lap_time_4 = time.perf_counter()

#run the function to filter the houses before going to asyncio
url_list_houses = house_filter_2(list_as_list(url_list))

lap_time_5 = time.perf_counter()

# Call the house_url_scraper function
#house_url_scraper(all_properties)
#with Session() as session:
#    all_houses = asyncio.run(house_url_scraper_async_main(all_properties))

with Session() as session:
    all_houses2 = asyncio.run(house_url_scraper_async_main(url_list_houses)) #asyncio with batches

#reformat the output of the asyncio/batch 
list = list_as_list(all_houses2)

#run the function to create a dataframe print it and save to csv
dataframe_to_csv(list, file_path = r'data\after_scraping.csv') # Give dataframe to save, and path to file

#Timer summary
end_time = time.perf_counter()
print(f"Stage 1 - knock_knock (no records yet): {lap_time_1 - start_time:.2f} seconds")
print(f"Stage 2 - overviewpage_counter (number of pages: {number_of_pages}): {lap_time_2 - start_time:.2f} seconds")
print(f"Stage 3 - url_list ({len(list_as_list(url_list)), type(url_list)}: {lap_time_3 - start_time:.2f} seconds")
#print(f"Stage 4 - extract_info_1 ({len(all_properties), type(all_properties)}: {lap_time_4 - start_time:.2f} seconds")
print(f"Stage 5 - house_filter 2 ({len(url_list_houses), type(url_list_houses)}: {lap_time_5 - start_time:.2f} seconds")
print(f"Stage 6 - house_url_scraper_async_main ({len(list), type(list)}): {end_time - start_time:.2f} seconds")
