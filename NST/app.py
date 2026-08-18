import os
import torch
from flask import Flask, render_template, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from PIL import Image
from torchvision import transforms

from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization, calc_mean_std

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
Bootstrap(app)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class UploadForm(FlaskForm):
    content = FileField('Content Image')
    style = FileField('Style Image')
    content_path = HiddenField()
    style_path = HiddenField()
    alpha = FloatField('Alpha', default=1.0)
    submit = SubmitField('Transfer Style')

# Render has no Apple MPS or NVIDIA GPU, so CPU is used there.
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Limit CPU thread usage on small Render instances.
if device.type == 'cpu':
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

VGG_PATH = os.path.join(BASE_DIR, 'vgg_normalised.pth')
DECODER_PATH = os.path.join(BASE_DIR, 'experiment', 'final_exp', 'decoder_final.pth')

if not os.path.isfile(VGG_PATH):
    raise FileNotFoundError(f'VGG model not found: {VGG_PATH}')
if not os.path.isfile(DECODER_PATH):
    raise FileNotFoundError(f'Decoder model not found: {DECODER_PATH}')

# Keep models unloaded during app startup. This prevents Gunicorn/Render
# health checks from blocking while large PyTorch weights are initialized.
encoder = None
decoder = None

def get_models():
    global encoder, decoder
    if encoder is None or decoder is None:
        encoder = VGGEncoder(VGG_PATH).to(device)
        decoder = Decoder().to(device)
        state = torch.load(DECODER_PATH, map_location=device)
        decoder.load_state_dict(state)
        del state
        encoder.eval()
        decoder.eval()
    return encoder, decoder

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def style_transfer(content_image, style_image, encoder, decoder, alpha, device):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor()
    ])
    content_image = transform(content_image).unsqueeze(0).to(device)
    style_image = transform(style_image).unsqueeze(0).to(device)

    with torch.no_grad():
        content_feats = encoder(content_image, is_test=True)
        style_feats = encoder(style_image, is_test=True)
        stylized_feats = adaptive_instance_normalization(content_feats, style_feats)
        stylized_feats = alpha * stylized_feats + (1 - alpha) * content_feats
        stylized_image = decoder(stylized_feats)

    return stylized_image

def save_image(image, path):
    image = image.detach().cpu().clone().squeeze(0).clamp(0, 1)
    transforms.ToPILImage()(image).save(path)

@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    result_image = None
    content_filename = None
    style_filename = None
    error = None

    if form.validate_on_submit():
        if form.content.data and form.content.data.filename:
            if allowed_file(form.content.data.filename):
                content_filename = secure_filename(form.content.data.filename)
                form.content.data.save(os.path.join(app.config['UPLOAD_FOLDER'], content_filename))
                form.content_path.data = content_filename
            else:
                error = 'Please upload a valid content image (PNG, JPG, or JPEG).'
        else:
            content_filename = form.content_path.data

        if form.style.data and form.style.data.filename:
            if allowed_file(form.style.data.filename):
                style_filename = secure_filename(form.style.data.filename)
                form.style.data.save(os.path.join(app.config['UPLOAD_FOLDER'], style_filename))
                form.style_path.data = style_filename
            else:
                error = 'Please upload a valid style image (PNG, JPG, or JPEG).'
        else:
            style_filename = form.style_path.data

        if not error and content_filename and style_filename:
            content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
            style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)
            try:
                content_image = Image.open(content_path).convert('RGB')
                style_image = Image.open(style_path).convert('RGB')
                alpha = float(form.alpha.data)
                encoder_model, decoder_model = get_models()
                stylized_image = style_transfer(
                    content_image, style_image, encoder_model, decoder_model, alpha, device
                )
                result_filename = 'stylized_' + content_filename
                result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
                save_image(stylized_image, result_path)
                result_image = result_filename
            except Exception as e:
                error = str(e)
        elif not error:
            missing = []
            if not content_filename:
                missing.append('content image')
            if not style_filename:
                missing.append('style image')
            error = 'Please upload ' + ' and '.join(missing) + '.'

    return render_template(
        'index.html', form=form, result_image=result_image,
        content_image=content_filename, style_image=style_filename, error=error
    )

@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/examples/<path:filename>')
def send_example(filename):
    return send_from_directory(os.path.join(BASE_DIR, 'examples'), filename)

if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('0.0.0.0', int(os.environ.get('PORT', 5000)), app, use_reloader=True, use_debugger=True)
