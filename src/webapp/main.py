import os
from app import app
import sys
import urllib.request
sys.path.insert(0, '..')
from src.caption_generator.caption_generator_inception import get_captions_inception
from src.caption_generator.caption_generator_vgg import get_captions_vgg
from src.caption_generator.caption_generator_resnet import get_captions_resnet
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # print('upload_image filename: ' + filename)
        inception_description = "Inception-V3 Caption: " + get_captions_inception(filename)
        # #description = inception_description
        # flash(inception_description)
        vgg16_description = "VGG-16 Caption:        " + get_captions_vgg(filename)
        resnet_description = "Resnet-50 Caption:    " + get_captions_resnet(filename)
        # flash(vgg16_description)
        flash(resnet_description)
        return render_template('upload.html', filename=filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    # print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


if __name__ == "__main__":
    app.run()