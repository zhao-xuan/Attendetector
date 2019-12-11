import requests
from time import sleep

url = 'http://localhost:5000/get_status'

while True:
    with requests.Session() as s:
        r = s.get(url)
        print(r.text)
    sleep(2)