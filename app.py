import base64
import glob
import logging
import sys
import uuid
import zipfile

from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField

import matplotlib
matplotlib.use('Agg')

import io
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import requests
import seaborn as sns
import shutil
import tempfile

from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, session, \
    make_response, jsonify, Response
from PIL import Image
CLIENT_SECRETS_FILE = 'credentials.json'
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
RANGE_PREFIX = 'Dec2018!'
RANGE_NAME = RANGE_PREFIX + 'A:Z'
DOG_IMAGE = 'dogimage.png'
DOG_MASK = 'dogmask.png'

pd.set_option('display.width', 1000)
pd.set_option('colheader_justify', 'center')

app = Flask(__name__)
# CORS(app)
app.config.from_pyfile('config.cfg')
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['CORS_SUPPORTS_CREDENTIALS'] = True
app.secret_key = 'dogfooding'

UPLOAD_FOLDER = os.path.join(app.instance_path, 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024  # 30 MB limit
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

sns.set()

DATASET_PATH = 'dogmentation_val.zip'
DATASET_IMAGE_COLUMN = 'image'
DATASET_MASK_COLUMN = 'mask'
BATCH_SIZE = 16

AUTHORIZATION = app.config.get('AUTHORIZATION')
URL = app.config.get('URL')
FIELD = 'mask'

with zipfile.ZipFile(DATASET_PATH, 'r') as z:
    index_file = z.open('index.csv')
    dataset = pd.read_csv(index_file)[:16]
    dataset[DATASET_IMAGE_COLUMN] = dataset[DATASET_IMAGE_COLUMN].apply(
        lambda path: Image.open(z.open(path))
    )
    dataset[DATASET_MASK_COLUMN] = dataset[DATASET_MASK_COLUMN].apply(
        lambda path: Image.open(z.open(path))
    )


def to_uploads(filename):
    return os.path.join(app.config['UPLOAD_FOLDER'], filename)


class ReusableForm(Form):
    name = StringField('Name:', validators=[validators.required()])
    email = StringField(
        'Email:',
        validators=[validators.required(),
                    validators.Length(min=6, max=35)])
    url = StringField(
        'URL:',
        validators=[validators.required(),
                    validators.Length(min=3, max=135)])
    token = StringField(
        'Token:',
        validators=[validators.required(),
                    validators.Length(min=3, max=135)])
    loss = StringField('Loss:', validators=[validators.required()])


def get_output_images(video_id, outdir, nr=3):
    files = glob.glob(os.path.join(outdir, f'{video_id}*.jpg'))
    # TODO Implement buckets for storage
    # Move to static directory for serving
    target_dir = app.static_folder
    local_files = []
    for file in files:
        target = to_uploads(os.path.basename(file))
        os.rename(file, target)
        local_files.append(''.join(target.split('/instance/')[-1]))
    # Get relative path
    total_files = len(local_files)
    if total_files <= nr:
        return files
    output_images = []
    for idx, interval_idx in enumerate(
            range(0, total_files, total_files // nr)):
        output_images.append(local_files[idx])
        if idx + 1 == nr:
            break
    return output_images


def credentials_to_dict(credentials):
    return {
        'token': credentials.token,
        'refresh_token': credentials.refresh_token,
        'token_uri': credentials.token_uri,
        'client_id': credentials.client_id,
        'client_secret': credentials.client_secret,
        'scopes': credentials.scopes
    }


@app.route('/reset')
def reset():
    session.clear()
    return redirect(url_for('/'))


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/clear_all')
def clear_all():
    try:
        os.remove(to_uploads('data.csv'))
        app.logger.info("Cleared dataframe")
    except FileNotFoundError:
        pass
    session.clear()
    return redirect('/')


@app.route('/remove/<row_idx>')
def remove(row_idx):
    filename = to_uploads('data.csv')
    print(f"Attempt to remove {row_idx}")
    df = pd.read_csv(to_uploads('data.csv'))
    row_idx = int(row_idx.split('button')[-1])

    try:
        df.drop(df.index[row_idx], inplace=True)
        if len(df) == 0:
            os.remove(filename)
        else:
            df.to_csv(filename, index=False)
    except:
        print(f"No index {row_idx} found in data.csv")
    return redirect('/', code=302)

@app.route('/backup_db')
def backup_db():
    from shutil import copyfile
    backup_filename = str(uuid.uuid4())+'.csv'
    copyfile(to_uploads('data.csv'), os.path.join(app.static_folder, backup_filename))
    return redirect('/', code=302)

def pil_to_bytearray(pil_img):
    img_byte_arr = io.BytesIO()
    pil_img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

def find_dogs(url, token, var_name, dog_image=DOG_IMAGE, debug=False):
    """Find dogs, returns PIL Image `out_img`."""
    test_dog = np.array(Image.open(dog_image))
    test_dog = test_dog.reshape(128, 128, 3).astype(np.float32)
    # if debug:
    # out_mask = Image.open(DOG_MASK)
    # else:
    out_mask = web_inference(test_dog, url, token,var_name)
    out_img = overlay_mask(dog_image, out_mask)
    out_img_filename = str(uuid.uuid4()) + '.png'
    out_img.save(to_uploads(out_img_filename))
    return out_img_filename

def web_inference(dog_image, url, token, var_name):
    """Run inference on dog_image, return grayscale PNG `out_mask`."""
    instance_path = os.environ.get('FLASK_INSTANCE_PATH')
    temp_filepath = os.path.join(instance_path, f'{uuid.uuid4()}.npy')
    np.save(temp_filepath, dog_image)
    headers = {'Authorization': 'Bearer ' + token}
    files = {'image': open(temp_filepath, 'rb')}
    response = requests.post(url, headers=headers, files=files).json()
    out_mask = response[var_name]
    # TODO: Convert to PIL for overlaying
    return out_mask

def overlay_mask(img_path: str, mask: Image):
    """Return Image"""
    img = Image.open(to_uploads(img_path))
    img = img.convert("RGBA")
    mask = decode_image(mask)
    mask = mask.convert("RGBA")
    overlayed_img = Image.blend(img, mask, alpha=.5)
    return overlayed_img

def evaluate_test_dataset(url, authorization):
    """Return list `rows` with responses for test `dataset`."""
    dataset['encoded'] = dataset[DATASET_IMAGE_COLUMN].apply(encode_image)

    if 'http' not in url:
        url = URL
    if len(authorization) < 10:
        authorization= AUTHORIZATION

    rows = []
    for i in range(0, len(dataset), BATCH_SIZE):
        data = {'rows': [
            {'image': encoded}
            for encoded in dataset['encoded'].iloc[i:(i + BATCH_SIZE)]
        ]}

        app.logger.debug(f"Sending request to {url}")
        response = requests.post(
            url,
            headers={'Authorization': authorization},
            json=data
        )
        rows.extend(response.json()['rows'])
    return rows

def encode_image(image: Image):
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='png')
    return 'data:image/png;base64,' + base64.b64encode(image_bytes.getvalue()).decode()


def decode_image(data_uri: str):
    format_str, image_str = data_uri.split(',', 1)
    if format_str != 'data:image/png;base64':
        raise ValueError('Format not supported')

    with io.BytesIO(base64.b64decode(image_str)) as image_bytes:
        image = Image.open(image_bytes)
        image.load()
        return image

def compute_iou(test_results_list):
    predicted_mask = np.stack(np.array(decode_image(row[FIELD])) for row in test_results_list)
    predicted_mask = np.sum(predicted_mask, axis=-1) > 255 / 2
    true_mask = np.stack(np.array(mask) for mask in dataset[DATASET_MASK_COLUMN])
    true_mask = true_mask > 255 / 2

    intersection = np.logical_and(predicted_mask, true_mask)
    union = np.logical_or(predicted_mask, true_mask)
    iou_score = np.sum(intersection) / np.sum(union)

    print('IoU', iou_score)
    return iou_score

def token_to_auth(token):
    if 'bearer' in token.lower():
        return token
    else:
        return 'Bearer ' + token

def save_img(img: Image):
    """"""
    img_path = str(uuid.uuid4())+'.png'
    img = decode_image(img)
    img.save(to_uploads(img_path))
    return img_path

def get_sample_overlay(test_results_list):
    """Return `out_img_filename` from list of dicts `test_results`."""
    mask = test_results_list[0][FIELD]
    img = dataset.loc[0,'encoded']
    img_path = save_img(img)
    overlay_img = overlay_mask(img_path, mask)
    out_img_path = str(uuid.uuid4())+'.png'
    overlay_img.save(to_uploads(out_img_path))
    return out_img_path

@app.route('/', methods=['GET', 'POST'])
def index():
    filename = to_uploads('data.csv')
    form = []
    results = []

    # results = get_results()
    if request.method == 'POST':
        form = ReusableForm(request.form)
        print(form.errors)
        name = request.form['name']
        if name is '':
            name = 'Anyonmous'
        email = request.form['email'].lower()
        url = request.form['url']
        auth = token_to_auth(request.form['token'])
        field_in = request.form['field_in'] or 'image'
        field_out = request.form['field_out'] or 'mask'
        # out_img_filename = find_dogs(url, auth, var_name, debug=True)
        test_results_list = evaluate_test_dataset(url, auth)
        out_img_filename = get_sample_overlay(test_results_list)

        data = {
            'timestamp': [pd.Timestamp.now()],
            'name': [name],
            'email': [email],
            'url': [url],
            'token': [auth],
            'field_in': [field_in],
            'field_out': [field_out],
            'out_img' : [out_img_filename],
            'iou' : 'NaN',
        }
        try:
            if os.path.exists(filename):
                data['iou'] = compute_iou(test_results_list)
                data['out_img'] = out_img_filename
                df = pd.read_csv(filename)
                df = df.append(pd.DataFrame(data, columns=[*data]))
            else:
                df = pd.DataFrame(data, columns=[*data])

            # Overwrite csv
            df.to_csv(filename,index=False)
            session['dataframe'] = df.to_html(
                # float_format=lambda x: '%.2f' % x,
                classes='mystyle')
            results = to_records(df)
        except TypeError:
            print("passing")
            pass

    if not results:
        try:
            results = pd.read_csv(filename)
            if results is not None:
                results = results.to_dict('records')
        except FileNotFoundError:
            app.logger.error("File not found")
        except TypeError:
            pass
    # import ipdb;ipdb.set_trace()
    print(results)

    return render_template('index.html', form=form, results=results)

def to_records(df, sortby=None):
    """Output `df` as dict-like `records` sorted by `sortby`."""
    if sortby is not None:
        df.sort_values(sortby, inplace=True)

    results = df.to_dict('records')
    return results

@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
else:
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
