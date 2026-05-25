from http.server import BaseHTTPRequestHandler
import json
import os
from datetime import datetime
from pymongo import MongoClient

MONGODB_URI = os.environ.get('MONGODB_URI')
client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
db = client['photo_album']
photos_col = db['photos']


def send_json(handler, data, status=200):
    handler.send_response(status)
    handler.send_header('Content-type', 'application/json')
    handler.send_header('Access-Control-Allow-Origin', '*')
    handler.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    handler.send_header('Access-Control-Allow-Headers', 'Content-Type')
    handler.end_headers()
    handler.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        try:
            docs = list(photos_col.find({}, {'_id': 0}))
            send_json(self, docs)
        except Exception as e:
            send_json(self, {'error': str(e)}, 500)

    def do_POST(self):
        try:
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length).decode('utf-8')
            data = json.loads(body)

            photo_id = str(int(datetime.now().timestamp() * 1000))
            photo = {
                'id': photo_id,
                'title': data.get('title', '未命名照片'),
                'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'imageBase64': data.get('imageBase64', ''),
                'comments': []
            }
            photos_col.insert_one(photo)
            # 去掉 _id 返回
            photo.pop('_id', None)
            send_json(self, photo, 201)
        except Exception as e:
            send_json(self, {'error': str(e)}, 500)
