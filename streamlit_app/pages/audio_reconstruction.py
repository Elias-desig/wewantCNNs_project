import streamlit as st
from core import load_model, reconstruction


model_options = st.selectbox("Select a Model", ["VAE", "Convolutional VAE", "Autoregressive NF"])
st.write(model_options)
instrument_source = st.selectbox('Instrument Source', ['acoustic', 'electronic', 'synthetic'])
instrument_family = st.selectbox('Instrument Familty', ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string', 'synth_lead', 'vocal'])
note_qualities = st.multiselect('Note Qualities', ['bright', 'dark', 'distortion', 'fast_decay', 'long_release', 'multiphonic', 'nonlinear_env', 'percussive', 'reverb', 'tempo-synced'])
st.write('Synthesize audio with Normalizing Flow:')
st.write('Synthesize audio with VAE:')