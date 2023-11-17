from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import numpy as np
import json
import glob
import os
import random
from eval import cache_model, generate_dance_for_music, g_http_path, g_upload_path


app = Flask(__name__)

visit_number = 0
@app.route("/")
def index():
    """
    global visit_number
    visit_number = visit_number + 1
    return "Hello, World %i" % visit_number
    """
    return render_template('upload_file.html')


@app.route('/file_upload', methods=['POST'])
def file_upload():
    file = request.files.get('file')
    if not file:
        return '上传失败，未选择文件'
    filename = secure_filename(file.filename)
    if '.' not in filename or filename.split('.')[-1] not in ['wav', 'mp3']:
        return '文件类型不支持'
    file.save(os.path.join(g_upload_path, file.filename))
    return '文件上传成功'


if __name__ == "__main__":
    cache_model()
    os.makedirs('./bvh/', exist_ok=True)
    app.run(host='0.0.0.0', port=5000)  # 127.0.0.1只能本机访问，0.0.0.0能局域网访问
