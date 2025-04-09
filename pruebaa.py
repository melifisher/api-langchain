import requests

response = requests.post('http://localhost:5000/api/initialize')
print(response.status_code)
print(response.text)
