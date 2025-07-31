import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np

import streamlit as st
from core import load_model, reconstruction
parent_dir = Path(__file__).parent.parent.parent
metadata_file = parent_dir /'data' / 'audio_samples' / 'nsynth-test/examples.json'

with open(metadata_file, 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data).T
df.columns = df.columns.astype(str)
model_options = st.selectbox("Select a Model", ["VAE", "Convolutional VAE", "Autoregressive NF"])
quality_options = ['bright', 'dark', 'distortion', 'fast_decay', 'long_release', 'multiphonic', 'nonlinear_env', 'percussive', 'reverb', 'tempo-synced']
st.header('Select a sample as a starting point:')
instrument_source = st.selectbox('Instrument Source', ['acoustic', 'electronic', 'synthetic'])
instrument_family = st.selectbox('Instrument Familty', ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string', 'synth_lead', 'vocal'])
note_qualities = st.multiselect('Note Qualities', quality_options)

st.write('Model:',model_options)
model = load_model(model_options)
results = df[
    (df['instrument_source_str'] == instrument_source) &
    (df['instrument_family_str'] == instrument_family)
]
if note_qualities:
    selected_idxs = [quality_options.index(q) for q in note_qualities]
    results = results[
        results['qualities']
               .apply(lambda bv: all(bv[i] == 1 for i in selected_idxs))
    ] 

def on_row_select(selected_idx):
    if selected_idx is None:
        return
    # get the DataFrame row and its index (the example ID)
    selected_row = results.iloc[selected_idx]
    example_id = selected_row.name

    st.write("Selected Example ID:", example_id)
    # reconstruct and play back
    reconstructed = reconstruction(
        model,
        str(parent_dir / 'data' / 'audio_samples' / f"{example_id}.wav")
    )
    st.audio(reconstructed, format='audio/wav')    

st.write("Results")
st.dataframe(results, selection_mode='single-row', on_select=)

