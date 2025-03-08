from bs4 import BeautifulSoup
import requests
url = 'https://www.tennisexplorer.com/indian-wells/2025/atp-men/'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
print(soup.prettify())