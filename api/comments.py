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
    handler.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
    handler.send_header('Access-Control-Allow-Headers', 'Content-Type')
    handler.end_headers()
    handler.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        try:
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length).decode('utf-8')
            data = json.loads(body)

            comment = {
                'name': data.get('name', '匿名用户')[:30],
                'text': data.get('text', '')[:500],
                'time': datetime.now().strftime('%m月%d日 %H:%M')
            }

            result = photos_col.update_one(
                {'id': data.get('photoId')},
                {'$push': {'comments': comment}}
            )

            if result.matched_count == 0:
                send_json(self, {'error': '照片不存在'}, 404)
                return

            send_json(self, {'success': True, 'comment': comment}, 201)
        except Exception as e:
            send_json(self, {'error': str(e)}, 500)
