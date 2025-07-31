import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import streamlit as st
from core import select_model, reconstruction, audio_to_melspectrogram
parent_dir = Path(__file__).parent.parent.parent
sys.path.append(str(parent_dir))
metadata_file = parent_dir /'data' / 'audio_samples' / 'nsynth-test/examples.json'

with open(metadata_file, 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data).T
df.columns = df.columns.astype(str)
model_options = st.selectbox("Select a Model (CVAE recommended)", ["VAE", "CVAE", "CVAE_Deep"])
quality_options = ['bright', 'dark', 'distortion', 'fast_decay', 'long_release', 'multiphonic', 'nonlinear_env', 'percussive', 'reverb', 'tempo-synced']
st.header('Select a sample as a starting point:')
instrument_source = st.selectbox('Instrument Source', ['acoustic', 'electronic', 'synthetic'])
instrument_family = st.selectbox('Instrument Familty', ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string', 'synth_lead', 'vocal'])
note_qualities = st.multiselect('Note Qualities', quality_options)

st.write('Model:',model_options)
model = select_model(model_options)
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

st.write("Results")
event = st.dataframe(
    results,
    selection_mode="single-row",
    on_select="rerun",
    key="recon_df"
)
if event and event.selection and event.selection.rows:
    selected_idx = event.selection.rows[0]
    selected_row = results.iloc[selected_idx]
    example_id = selected_row.name
    st.write("Selected Example ID:", example_id)
    path = str(parent_dir / 'data' / 'audio_samples' / 'nsynth-test' / 'audio' / f"{example_id}.wav")
    st.audio(path, format='audio/wav')
    fig, ax = plt.subplots()
    ax.imshow(audio_to_melspectrogram(path))
    st.pyplot(fig)    
    audio, reconstructed, latent_z = reconstruction(model, model_options,
    path    ,
        model_options == 'CVAE'
    )
    st.write('Audio reconstruction')
    st.audio(audio, format="audio/wav", sample_rate=16000)
    fig, ax = plt.subplots()
    ax.imshow(reconstructed)
    st.pyplot(fig)
