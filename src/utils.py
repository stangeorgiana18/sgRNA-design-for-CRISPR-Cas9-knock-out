# Helper functions (API, validation)

import requests
from Bio.Seq import Seq
import re

def fetch_gene_sequence(gene_name: str) -> str:
    """
    Fetch the genomic DNA sequence of a gene from Ensembl.
    Args: gene_name (str): Gene symbol (e.g., "TP53").
    Returns: str: DNA sequence (uppercase, no spaces).
    Raises: ValueError: If gene not found or API fails.

    """

    # Step 1: Get Ensembl gene ID from the gene symbol 
    # Convert gene name to uppercase for consistency
    # Standardizing gene names to uppercase avoids ambiguity (e.g., tp53 vs TP53).
    gene_name_upper = gene_name.upper()
    lookup_url = f"https://rest.ensembl.org/lookup/symbol/human/{gene_name_upper}?expand=1"
    response = requests.get(lookup_url, headers={"Content-Type": "application/json"})
    
    if response.status_code != 200:
        raise ValueError(f"Gene {gene_name_upper} not found in Ensembl.")
    
    gene_data = response.json()
    gene_id = gene_data["id"]
    
    # Step 2: Fetch the genomic sequence
    sequence_url = f"https://rest.ensembl.org/sequence/id/{gene_id}?type=genomic"
    response = requests.get(sequence_url, headers={"Content-Type": "text/plain"})
    
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch sequence for {gene_name}.")
    
    sequence = response.text.strip().upper()
    
    # Step 3: Validate sequence
    if not re.match("^[ATCG]*$", sequence):
        raise ValueError(f"Sequence for {gene_name} contains non-DNA characters.")
    
    if len(sequence) < 23:
        # If Ensembl returns a short sequence (rare for genes), it will return:
        raise ValueError(f"Sequence for {gene_name} is too short (min 23 bp required).")
    
    return sequence


def validate_dna_sequence(sequence: str) -> bool:
    """
    Check if a sequence contains only valid DNA characters (A, T, C, G).
    """
    return re.fullmatch("^[ATCG]*$", sequence) is not None


def validate_sequence_length(sequence: str) -> bool:
    """
    Check if the sequence meets the minimum length requirement (23 bp).
    """
    return len(sequence) >= 23