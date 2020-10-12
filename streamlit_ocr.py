# Core Pkgs

from PIL import Image
import numpy as np 
import streamlit as st 
st.set_option('deprecation.showfileUploaderEncoding', False)

from modelOCR import numericalDetectron2, boundingBoxesDetectron2, alphabeticalDetectron2, easypredict
from config import cfg

@st.cache
def load_models():
    bBoxDet = boundingBoxesDetectron2(cfg['boundingBoxesDetectron2'])
    numDet = numericalDetectron2(cfg['numericalDetectron2'])
    alphaDet = alphabeticalDetectron2(cfg['alphabeticalDetectron2'])
    return bBoxDet, numDet, alphaDet

@st.cache
def predict_ktp(image):
    img_array = np.array(image)
    res = easypredict(img_array, bBoxDet, numDet, alphaDet, input_type='ktp')
    del res['input_type']
    res = "\n".join([str(k) + " \t : " + str(v) for k,v in res.items()])
    return res

@st.cache
def predict_sim(image):
    img_array = np.array(image)
    res = easypredict(img_array, bBoxDet, numDet, alphaDet, input_type='sim')
    del res['input_type']
    res = "\n".join([str(k) + " \t : " + str(v) if k != 'expdate' else str(k) + "  : " + str(v) for k,v in res.items()])
    return res

bBoxDet, numDet, alphaDet = load_models()


st.title("OCR Astra Digital")
document_type = ["KTP","SIM"]
document_choice = st.selectbox("Select Document Type",document_type)

image_file = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])
if image_file is not None:
    img = Image.open(image_file)
    st.image(img, use_column_width=True)
    
    if document_choice == "KTP":
        res = predict_ktp(img)
        st.text(res)
        
    elif document_choice == "SIM":
        res = predict_sim(img)
        st.text(res)
        
"ZIYAD"
