import model
import requests

url = "http://127.0.0.1:5000/predict"
params = {"time" : 10,
          "ejection_fraction" : 22,
          "serum_creatinine" : 0.1,
          "age" : 35
          }

req = requests.post(url = url, data = params)
# print(req.json())
