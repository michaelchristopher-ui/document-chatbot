"""Entrypoint: `streamlit run app.py`."""

from dotenv import load_dotenv

from adapters.inbound.streamlit_ui import main

load_dotenv()
main()
