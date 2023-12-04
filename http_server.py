from flask import Flask, request, render_template, make_response
from werkzeug.utils import secure_filename
import numpy as np
import json
import glob
import os
import random
import json
from eval import cache_all_models, generate_action_for_audio, get_model_config, bvh2uedata
import codecs


app = Flask(__name__)


@app.route("/")
def index():
    #return render_template('upload_file.html')
    """
    style = request.args.get("style")
    if style is None or style not in ['gOK', 'gFF']:
        error = f'style is invalid'
        return error
    return render_template(f'upload_and_generate_{style}.html')
    """
    return render_template(f'upload_and_generate.html')


@app.route("/test")
def test():
    response = make_response("test", 200, {'name': 'test'})
    return response


@app.route('/file_upload', methods=['POST'])
def file_upload():
    file = request.files.get('file')
    if not file:
        return 'no file chosen'
    filename = file.filename
    #filename = secure_filename(file.filename)
    if '.' not in filename or filename.split('.')[-1] not in ['wav', 'mp3']:
        return 'file ext not supported'
    #file.save(os.path.join('./http_data/upload', filename))

    encoder = codecs.getincrementalencoder('utf-8')()
    filename = encoder.encode(filename)
    content = filename
    response = make_response(content, 200)
    response.headers['Content-Type'] = 'text/plain;charset=UTF-8'
    response.headers['audio'] = filename
    return response

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
        error, motion_data = generate_action_for_audio(music_name)
        #json_data = {'error': error, 'motion': motion_data}
    return motion_data


@app.route("/upload_and_generate", methods=['POST'])
def upload_and_generate():
    """
    style = request.args.get("style")
    start = request.args.get("start")
    """
    file = request.files.get('file')
    if not file:
        error = 'no file chosen'
        print(error)
        return error

    filename = file.filename
    #filename = secure_filename(file.filename)
    if filename is not None:
        print(f'upload_and_generate(), received {file.filename}, {filename}')
    if '.' not in filename or filename.split('.')[-1] not in ['wav', 'mp3']:
        error = 'file ext not supported'
        print(error)
        return error

    basename = filename.split('.')[0]
    style = basename.split('_')[-1]
    if style is None:
        style = 'gFF'
    model_config = get_model_config(style)
    if model_config is None:
        style = 'gFF'
        model_config = get_model_config(style)

    upload_path = model_config['upload_path']
    file.save(os.path.join(upload_path, filename))
    print(f'Uploaded file {filename}')

    start = 30
    error, json_data = generate_action_for_audio(filename, style_token=style, start_seconds=start)
    if error is not None:
        content = error
        response = make_response(content, 200)
    else:
        encoder = codecs.getincrementalencoder('utf-8')()
        filename = encoder.encode(filename)  # json_data['audio']
        content = json.dumps(json_data['motion'])
        response = make_response(content, 200)
        response.headers['audio'] = filename
    return response


if __name__ == "__main__":
    # nohup python -u http_server.py > loghttp.log 2>&1 &
    cache_all_models()
    os.makedirs('./bvh/', exist_ok=True)
    app.run(host='0.0.0.0', port=5000)  # 127.0.0.1只能本机访问，0.0.0.0能局域网访问
