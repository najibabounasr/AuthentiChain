pragma solidity ^0.5.0;

import "@openzeppelin/contracts/token/ERC721/ERC721Full.sol";

// Define the contract
contract AuthentiChain is ERC721Full {
    // Set the name and symbol for the token in the constructor
    constructor() public ERC721Full("AuthentiChain", "ATC") {}

    // Function to mint new tokens
    // `msg.sender` is the address that calls this function
    function mint(uint256 tokenId) public {
        // call the internal mint function from the ERC721 contract
        _mint(msg.sender, tokenId);
    }

    // Function to set the token URI (where the metadata is stored)
    function setTokenURI(uint256 tokenId, string memory tokenURI) public {
        // call the internal setTokenURI function from the ERC721 contract
        _setTokenURI(tokenId, tokenURI);
    }

    // Function to transfer tokens from one address to another
    function transfer(address from, address to, uint256 tokenId) public {
        // call the internal transfer function from the ERC721 contract
        _transferFrom(from, to, tokenId);
    }
}


