import requests
import json
from flask import Flask, request
app = Flask(__name__)


@app.route('/receive', methods=['POST'])
def receive_data():
    data = request.get_json()  # 获取POST请求中的JSON数据
    print("Received data:", data)
    return "Data received successfully"
if __name__ == "__main__":
    url = 'http://127.0.0.1:5000/tasks'
    #response = requests.post('http://127.0.0.1:5000/picture', json={'base64':'https://th.bing.com/th/id/R.fa2c64d3e2c99198337cd2f9a3baf31a?rik=K%2f4wkmiEsxPMaA&riu=http%3a%2f%2fpic12.nipic.com%2f20110111%2f2457331_001350942000_2.jpg&ehk=CTdSDuFkIJ9GFTkwES58mDEch7beKNKSF1oznskoyJQ%3d&risl=&pid=ImgRaw&r=0',
                                                                     #'analysisRule' : {}})
    #response = requests.post('http://127.0.0.1:5000/tasks', json={"taskId": "1", 'type': '1', 'url': '0', 'analysisRule' : {'objType' : [22001]}})
    #response = requests.post('http://127.0.0.1:5000/pictures')
    #response = requests.delete('http://127.0.0.1:5000/tasks/1')
    #response = requests.get('http://127.0.0.1:5000/tasks')
    response = requests.post(url, json={"taskId": "2", 'type': '1', 'url': 'http://admin:admin@192.168.1.111:8081', 'analysisRule' : {}})
    print(response.json())
    app.run(host="0.0.0.0", port=7000)