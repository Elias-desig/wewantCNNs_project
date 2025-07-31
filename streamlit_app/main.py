# main.py (single-page controller)
import streamlit as st
from pathlib import Path 
import sys
base_dir = Path(__file__).parent.parent
sys.path.append(str(base_dir))

from core import unpack_jsonwav_archive

audio_file = base_dir / 'data' / 'nsynth-test.jsonwav.tar.gz'
audio_dir = base_dir / 'data' / 'audio_samples'

if not audio_dir.exists():
    st.info('Extracting Audio Files...')
    unpack_jsonwav_archive(audio_file, audio_dir)
    st.success('Audio Files ready!')


pages = {'Model Architectures':
         [st.Page("pages/vae_architecture.py", title='VAEs'),
          st.Page("pages/nf_architecture.py", title='Normalizing Flows')],
          'Generate Audio':
          [st.Page("pages/audio_reconstruction.py", title='Audio Reconstruction'),
           st.Page("pages/latent_interface.py", title='Latent Space Interfae')]}



page = st.navigation(pages, position='top')
page.run()

