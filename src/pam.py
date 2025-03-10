# Logic for PAM/sgRNA

from Bio.Seq import Seq # used to reverse-complement sequences for the reverse strand
from typing import List, Tuple
import re

"""
pam.py: Functions for PAM detection and sgRNA candidate extraction and filtering.

This module provides functionality to:
  - Search for PAM motifs in a given DNA sequence (both forward and reverse strands).
  - Extract candidate sgRNA sequences (typically 20bp guides upstream of the PAM for SpCas9).
  - Filter candidate sgRNAs based on design rules (GC content, homopolymers, self-complementarity).

Functions:
    find_pam_sites(sequence: str, pam: str = "NGG", strand: str = "both") -> List[Tuple[str, int, int, str]]
    valid_gc(sgRNA: str, min_gc: float = 0.4, max_gc: float = 0.6) -> bool
    no_homopolymers(sgRNA: str, max_repeats: int = 3) -> bool
    no_self_complementarity(sgRNA: str, seed_length: int = 8) -> bool
    filter_sgrnas(sgrnas: List[Tuple[str, int, int, str]]) -> List[Tuple[str, int, int, str]]
"""

from Bio.Seq import Seq
from typing import List, Tuple
import re

def find_pam_sites(sequence: str, pam: str = "NGG", strand: str = "both") -> List[Tuple[str, int, int, str]]:
    """
    Find PAM sites in a DNA sequence and extract candidate sgRNA sequences.
    
    For the forward strand (if strand is "forward" or "both"):
      - Searches for the PAM motif in the original sequence.
      - Extracts the 20bp immediately upstream of the PAM (if available).
    
    For the reverse strand (if strand is "reverse" or "both"):
      - Computes the reverse complement of the sequence.
      - Searches for the PAM motif in the reverse complement.
      - Extracts the 20bp immediately upstream (in the reverse complement) of the PAM.
      - Converts the coordinates back to the original (forward) sequence.
    
    Args:
        sequence (str): Input DNA sequence.
        pam (str): PAM motif (default "NGG"). 'N' is treated as a wildcard.
        strand (str): Which strand(s) to search: "forward", "reverse", or "both".
    
    Returns:
        List[Tuple[str, int, int, str]]: A list of candidate sgRNAs.
            Each tuple is (sgRNA, start, end, strand), where:
              - sgRNA is the 20-bp candidate guide,
              - start and end are 0-based coordinates on the original sequence,
              - strand is "+" for forward and "-" for reverse.
    """
    sequence = sequence.upper()
    pam_pattern = pam.upper().replace("N", "[ATCG]")
    candidates = []
    seq_len = len(sequence)
    
    # Forward strand: sgRNA is the 20bp immediately upstream of the PAM.
    if strand in ["forward", "both"]:
        for match in re.finditer(pam_pattern, sequence):
            pam_start = match.start()  # PAM starts at this index.
            sgRNA_start = pam_start - 20
            if sgRNA_start < 0:
                continue  # Not enough room for a full guide.
            sgRNA_seq = sequence[sgRNA_start:pam_start]
            if len(sgRNA_seq) == 20:
                candidates.append((sgRNA_seq, sgRNA_start, pam_start, "+"))
    
    # Reverse strand: work on the reverse complement.
    if strand in ["reverse", "both"]:
        rev_sequence = str(Seq(sequence).reverse_complement())
        for match in re.finditer(pam_pattern, rev_sequence):
            pam_start_rc = match.start()  # PAM start in reverse complement.
            sgRNA_start_rc = pam_start_rc - 20
            if sgRNA_start_rc < 0:
                continue
            sgRNA_rc = rev_sequence[sgRNA_start_rc:pam_start_rc]
            # Convert the candidate sgRNA back to the forward orientation.
            sgRNA_seq = str(Seq(sgRNA_rc).reverse_complement())
            # Convert reverse complement coordinates to original coordinates.
            # In rev_sequence, index i corresponds to original index: (seq_len - 1 - i).
            orig_end = seq_len - sgRNA_start_rc  # End of sgRNA in original.
            orig_start = seq_len - pam_start_rc   # Start of sgRNA in original.
            candidates.append((sgRNA_seq, orig_start, orig_end, "-"))
    
    return candidates

def valid_gc(sgRNA: str, min_gc: float = 0.4, max_gc: float = 0.6) -> bool:
    """
    Check if the sgRNA has an acceptable GC content.
    
    Args:
        sgRNA (str): The sgRNA sequence.
        min_gc (float): Minimum acceptable GC content (default 0.4).
        max_gc (float): Maximum acceptable GC content (default 0.6).
    
    Returns:
        bool: True if the GC content is within the range, False otherwise.
    """
    gc_count = sum(1 for base in sgRNA if base in "GC")
    gc_content = gc_count / len(sgRNA)
    return min_gc <= gc_content <= max_gc

def no_homopolymers(sgRNA: str, max_repeats: int = 3) -> bool:
    """
    Ensure the sgRNA does not contain homopolymers longer than allowed.
    
    Args:
        sgRNA (str): The sgRNA sequence.
        max_repeats (int): Maximum allowed consecutive identical bases (default 3).
    
    Returns:
        bool: True if no homopolymers exceed the limit, False otherwise.
    """
    for base in "ATCG":
        if re.search(base * (max_repeats + 1), sgRNA):
            return False
    return True

def no_self_complementarity(sgRNA: str, seed_length: int = 8) -> bool:
    """
    Check that the sgRNA does not have self-complementarity in the seed region.
    
    The seed region is defined as the last `seed_length` bases of the sgRNA.
    The function checks that the reverse complement of the seed region does not appear
    elsewhere in the sgRNA.
    
    Args:
        sgRNA (str): The sgRNA sequence.
        seed_length (int): Length of the seed region to check (default 8).
    
    Returns:
        bool: True if no self-complementarity is found, False otherwise.
    """
    if len(sgRNA) < seed_length:
        return True
    seed = sgRNA[-seed_length:]
    seed_rc = str(Seq(seed).reverse_complement())
    return seed_rc not in sgRNA[:-seed_length]

def filter_sgrnas(sgrnas: List[Tuple[str, int, int, str]]) -> List[Tuple[str, int, int, str]]:
    """
    Filters sgRNA candidates based on design rules:
      - GC content must be between 40% and 60%
      - No homopolymers longer than allowed (default > 3 consecutive bases)
      - No self-complementarity in the seed region (default seed length 8)
    
    Args:
        sgrnas (List[Tuple[str, int, int, str]]): List of candidate sgRNAs as tuples (sgRNA, start, end, strand).
    
    Returns:
        List[Tuple[str, int, int, str]]: Filtered list of sgRNA candidates.
    """
    filtered = []
    for sgRNA, start, end, strand in sgrnas:
        if not valid_gc(sgRNA):
            continue
        if not no_homopolymers(sgRNA):
            continue
        if not no_self_complementarity(sgRNA):
            continue
        filtered.append((sgRNA, start, end, strand))
    return filtered

if __name__ == "__main__":
    # Example usage: test the functions on a sample sequence.
    example_sequence = "ATGCGTACGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC"
    example_sequence = example_sequence[:50]
    
    print("Finding candidate sgRNAs (both strands) in the example sequence:")
    candidates = find_pam_sites(example_sequence, pam="NGG", strand="both")
    for candidate in candidates:
        print(candidate)
    
    print("\nFiltered candidate sgRNAs (after applying design rules):")
    filtered_candidates = filter_sgrnas(candidates)
    for candidate in filtered_candidates:
        print(candidate)
