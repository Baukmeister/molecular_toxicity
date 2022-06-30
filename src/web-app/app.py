import os.path
from base64 import *
import io

from PIL import Image
from flask import Flask, request, jsonify, render_template
from rdkit import Chem
from rdkit.Chem.Draw import MolToImage

# create the flask app
app = Flask(__name__)


# what html should be loaded as the home page when the app loads?
@app.route('/')
def home():
    return render_template('home.html', prediction_text="")


# define the logic for reading the inputs from the WEB PAGE,
# running the model, and displaying the prediction
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    smiles = request.form.get('smiles')
    molecule_image = _draw_smiles(smiles)
    dataurl = 'data:image/png;base64,' + molecule_image
    return render_template('home.html', prediction_text="Toxic", image_data=dataurl)


def _draw_smiles(smiles_string):
    file_name = f"images/{smiles_string}.png"

    if not os.path.exists(file_name):
        molecule = Chem.MolFromSmiles(smiles_string)
        img = MolToImage(molecule)
        img.save(file_name)

    img = Image.open(file_name)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    my_encoded_img = encodebytes(img_byte_arr.getvalue()).decode('ascii')

    return my_encoded_img

if __name__ == "__main__":
    app.run(debug=True)
