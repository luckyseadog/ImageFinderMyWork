import streamlit as st
from model import DummyIndexer, EmbedderCLIP, SearchModel
import re
import sys
from attention_map import *

path = '/Users/vas/Desktop/custom_datasett'
clip_model = SearchModel(EmbedderCLIP(device='cpu'), DummyIndexer())
clip_model.load_imgs(path)
clip_model.save_embs()


st.title('CLIP model')
request = st.text_input('Search for')
k = st.slider('How many pictures to  show?', 0, 10, 3)

if st.button('Go!'):
    if re.fullmatch('[a-zA-Z]+', request):
        query = clip_model.embedder.encode_text(text=request)
        d, urls = clip_model.get_k_imgs(query, k)
        for url in urls:
            saliency_layer = "layer4"
            blur = True

            image_input = clip_model.embedder.preprocess(Image.open(url)).unsqueeze(0)
            image_np = load_image(url, clip_model.embedder.predictor.visual.input_resolution)
            text_input = clip.tokenize([request])

            attn_map = gradCAM(
                clip_model.embedder.predictor.visual,
                image_input,
                clip_model.embedder.predictor.encode_text(text_input).float(),
                getattr(clip_model.embedder.predictor.visual, saliency_layer)
            )
            attn_map = attn_map.squeeze().detach().cpu().numpy()

            st.image([Image.open(str(url)), getAttMap(image_np, attn_map, blur)], clamp=True)
    else:
        st.info("Error, please use query in English")





