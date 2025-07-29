# main.py (single-page controller)
import streamlit as st



pages = {'Model Architectures':
         [st.Page("pages/vae_architecture.py", title='VAEs'),
          st.Page("pages/nf_architecture.py", title='Normalizing Flows')],
          'Generate Audio':
          [st.Page("pages/audio_reconstruction.py", title='Audio Reconstruction'),
           st.Page("pages/latent_interface.py", title='Latent Space Interfae')]}



page = st.navigation(pages, position='top')
page.run()

