import streamlit as st
from web3 import Web3
import uuid
import qrcode
from funcs.streamlit_functions import generate_unique_identifier, generate_qr_code
from dotenv import load_dotenv



# Load environment variables
load_dotenv()

# Load Ethereum smart contract ABI and address (replace with your own values)
CONTRACT_ABI = 'your_smart_contract_abi'
CONTRACT_ADDRESS = 'your_smart_contract_address'

# Streamlit UI
st.title("Manufacturer")
st.subheader("""



""")


with st.form(key="product_form"):
    product_name = st.text_input("Product Name")
    st.session_state['product_name'] = product_name
    product_desc = st.text_input("Product Description")
    st.session_state['product_desc'] = product_desc
    submit_button = st.form_submit_button("Generate QR Code")
    st.session_state['submit_button'] = submit_button

if submit_button:
    unique_identifier = generate_unique_identifier(product_name, product_desc)
    st.session_state['unique_identifier'] = unique_identifier
    qr_image = generate_qr_code(unique_identifier)
    st.session_state['qr_image'] = qr_image
    st.image(qr_image, caption="Generated QR Code", use_column_width=True)
