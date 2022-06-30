import os.path
from base64 import *
import io

import torch
from PIL import Image
from flask import Flask, request, jsonify, render_template
from rdkit import Chem
from rdkit.Chem.Draw import MolToImage

# create the flask app
from src.molecule_parsing.molecule_data_handler import MoleculeDataHandler

app = Flask(__name__)


# what html should be loaded as the home page when the app loads?
@app.route('/')
def home():
    return render_template('home.html', prediction_text="")


# define the logic for reading the inputs from the WEB PAGE,
# running the model, and displaying the prediction
@app.route('/predict-toxicity', methods=['GET', 'POST'])
def predict():
    smiles = request.form.get('smiles')
    molecule_image = _draw_smiles(smiles)
    toxicity_prediction = _predict_molecule_tox(smiles)
    dataurl = 'data:image/png;base64,' + molecule_image
    return render_template('home.html', prediction_text=toxicity_prediction, image_data=dataurl,
                           pred_color="green" if toxicity_prediction == 'Non-Toxic' else "red")


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


def _predict_molecule_tox(smiles_string):
    handler = MoleculeDataHandler(load_cache=False)
    _, graph = handler.parse_smiles_molecule(smiles_string)
    pyG_graph = handler.convert_molecules_to_pyG(graph)
    trained_model = torch.load("../trained_model/best_model.pt")
    result = int(trained_model(pyG_graph).argmax(dim=1))

    return "Toxic" if result == 1 else "Non-Toxic"


if __name__ == "__main__":
    app.run(debug=True)
