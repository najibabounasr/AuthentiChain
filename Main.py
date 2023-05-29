import streamlit as st
from web3 import Web3
import uuid
import qrcode
from funcs.streamlit_functions import generate_unique_identifier
from funcs.streamlit_functions import generate_qr_code, verify_product
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load Ethereum smart contract ABI and address (replace with your own values)
CONTRACT_ABI = 'your_smart_contract_abi'
CONTRACT_ADDRESS = 'your_smart_contract_address'


# Streamlit UI
st.title("AuthentiChain")
# Explain the purpose of the app
st.subheader("""
Utilizing blockchain technology to verify the authenticity of products in a supply chain has ben a popular use case for blockchain. This app demonstrates how to build a supply chain app using blockchain and Python.
Some examples of larger blockchain-based supply chain management systems include Amazon Managed Blockchain, IBM Food Trust, and Oracle Blockchain Platform. All these utilize 'track and trace', the ability to track all past and present locations of product inventory.
If implemented early on in the supply chain, blockchain can be used to track the origin of raw materials, and the conditions in which they were produced. This can be used to ensure that the product was produced in a sustainable manner, and that the product is authentic.

In a world wholly dependant on international trade, supply-chain management and authenticicity verification have become ciritical concerns across multiple industries. The 2021 global sypply chain crisis revealed to the world the fragility of the global supply chain,
and the need for more robust supply chain management systems. The blockchain technology can very easily pinpoint where in the supply chain a product is, and if lost, which specific product is lost, and where it had ben lost. The QR code technology is only a stepping stone
to more robust identifiers. 
""")
# Explain the purpose of the app           
st.warning("This application serves to demonstrate the capabilities of blockchain ledger technology, and is not meant to be used in a production environment.")

# Developer information
developers = [
    {
        "name": "Najib Abou Nasr",
        "socials": {
            "LinkedIn": "https://www.linkedin.com/in/najib-abou-nasr-a43520258/",
            "GitHub": "https://github.com/najibabounasr",
            "Twitter": "https://twitter.com/najib_abounasr"
        },
        "info": "Current Computer Science Major at Santa Monica College. Interested in web3 technologies, and its applications in the real world. "
    },
    {
        "name": "Alex Lichcaeva",
        "socials": {
            "LinkedIn": "https://www.linkedin.com/developer2",
            "GitHub": "https://github.com/developer2",
            "Twitter": "https://twitter.com/developer2"
        },
        "info": "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
    },
    {
        "name": "Alphonso Logan",
        "socials": {
            "LinkedIn": "https://www.linkedin.com/developer3",
            "GitHub": "https://github.com/developer3",
            "Twitter": "https://twitter.com/developer3"
        },
        "info": "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
    }
]

# Display developer information
st.sidebar.title("Developers")
for developer in developers:
    st.sidebar.subheader(developer["name"])
    st.sidebar.markdown(developer["info"])
    for social, link in developer["socials"].items():
        st.sidebar.markdown(f"[{social}]({link})")

    st.sidebar.write("---")


