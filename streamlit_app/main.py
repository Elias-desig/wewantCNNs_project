# main.py (single-page controller)
import streamlit as st
from pathlib import Path 
import sys

st.set_page_config(
    page_title="WeWantCNNs Audio Generation",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)


base_dir = Path(__file__).parent.parent
sys.path.append(str(base_dir))

from core import unpack_jsonwav_archive

def main_page():
    st.title('WeWantCNNs Audio Generation Project')
    st.markdown(""" Welcome to the WeWantCNNs website! We imlemented two different types of probabilistic DNN architectures for audio synthesis. 
                """)
    col1, col2 = st.columns(2)
    with col1:
        st.info("✅ VAE Models Available")
        st.info("🔄 Normalizing Flows In Development")
    with col2:
        st.info("✅ Audio Reconstruction Ready")
        st.info("🔄 Latent Interface Coming Soon")    


@st.cache_data
def initialize_audio_data():
    audio_file = base_dir / 'data' / 'nsynth-test.jsonwav.tar.gz'
    audio_dir = base_dir / 'data' / 'audio_samples'

    if not audio_dir.exists():
        st.info('Extracting Audio Files...')
        unpack_jsonwav_archive(audio_file, audio_dir)
        st.success('Audio Files ready!')
    return True
initialize_audio_data()

pages = {
    'Home': [st.Page(main_page, title='Home')],
    'Model Architectures':
         [st.Page("pages/vae_architecture.py", title='VAEs'),
          st.Page("pages/nf_architecture.py", title='Normalizing Flows')],
          'Generate Audio':
          [st.Page("pages/audio_reconstruction.py", title='Audio Reconstruction'),
           st.Page("pages/latent_interface.py", title='Latent Space Interfae')]}



page = st.navigation(pages, position='top')
page.run()

