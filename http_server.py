from flask import Flask, request, render_template, make_response
from werkzeug.utils import secure_filename
import numpy as np
import json
import glob
import os
import random
import json
from eval import cache_model, generate_dance_for_music, bvh2uedata, g_upload_path


app = Flask(__name__)

visit_number = 0
@app.route("/")
def index():
    """
    global visit_number
    visit_number = visit_number + 1
    return "Hello, World %i" % visit_number
    """
    #return render_template('upload_file.html')
    return render_template('upload_and_generate.html')

@app.route("/test")
def test():
    response = make_response("test", 200, {'name': 'test'})
    return response


@app.route('/file_upload', methods=['POST'])
def file_upload():
    file = request.files.get('file')
    if not file:
        return 'no file chosen'
    filename = secure_filename(file.filename)
    if '.' not in filename or filename.split('.')[-1] not in ['wav', 'mp3']:
        return 'file ext not supported'
    file.save(os.path.join(g_upload_path, file.filename))
    return 'success'


@app.route("/get_motion")
def get_motion():
    DEBUG = True
    music_name = request.args.get("music")
    if music_name is None:
        if DEBUG:
            error = 'OK'
            ue_data = bvh2uedata('./http_data/debug.bvh')
            motion_data = json.dumps(ue_data)
        else:
            motion_data = ''
            #json_data = {'error': 'arg error'}
    else:
        music_name = secure_filename(music_name)
        error, motion_data = generate_dance_for_music(music_name)
        #json_data = {'error': error, 'motion': motion_data}
    return motion_data

@app.route("/upload_and_generate", methods=['POST'])
def upload_and_generate():
    file = request.files.get('file')
    if not file:
        error = 'no file chosen'
        print(error)
        return error
    filename = secure_filename(file.filename)
    if filename is not None:
        print(f'upload_and_generate(), received {file.filename}, {filename}')
    if '.' not in filename or filename.split('.')[-1] not in ['wav', 'mp3']:
        error = 'file ext not supported'
        print(error)
        return error
    file.save(os.path.join(g_upload_path, file.filename))

    error, json_data = generate_dance_for_music(filename)
    if error is not None:
        content = error
        response = make_response(content, 200)
    else:
        content = json.dumps(json_data['motion'])
        response = make_response(content, 200)
        response.headers['audio'] = json_data['audio']
    return response


if __name__ == "__main__":
    cache_model()
    os.makedirs('./bvh/', exist_ok=True)
    app.run(host='0.0.0.0', port=5000)  # 127.0.0.1只能本机访问，0.0.0.0能局域网访问
