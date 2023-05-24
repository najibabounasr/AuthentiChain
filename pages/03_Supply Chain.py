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
st.title("Supply Chain")
