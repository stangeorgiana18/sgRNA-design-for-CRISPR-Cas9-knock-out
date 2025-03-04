# Logic for PAM/sgRNA

from Bio.Seq import Seq # used to reverse-complement sequences for the reverse strand
from typing import List, Tuple
import re

def find_pam_sites(sequence: str, pam: str = "NGG", strand: str = "both") -> List[Tuple[str, int, str]]:
    """
    Find PAM sites in a DNA sequence and extract sgRNA candidates.
    
    Args:
        sequence: Input DNA sequence.
        pam: PAM motif (e.g., 'NGG' for SpCas9).
        strand: 'forward', 'reverse', or 'both'.
    
    Returns:
        List of (sgRNA, position, strand) tuples.
    """
    sequence = sequence.upper()
    pam = pam.upper().replace("N", "[ATCG]")  # Convert "NGG" to regex "[ATCG]GG"
    candidates = []
    
    # Scan forward strand
    if strand in ["forward", "both"]:
        for match in re.finditer(pam, sequence):
            start = match.start() - 20  # PAM starts at position 20 of the sgRNA
            if start >= 0:
                sgRNA = sequence[start:start+20]
                candidates.append((sgRNA, start, "+"))
    
    # Scan reverse strand
    if strand in ["reverse", "both"]:
        rev_sequence = str(Seq(sequence).reverse_complement())
        for match in re.finditer(pam, rev_sequence):
            start = match.start() - 20
            if start >= 0:
                sgRNA = rev_sequence[start:start+20]
                # Convert reverse strand positions to original sequence coordinates
                orig_pos = len(sequence) - (start + len(pam) + 1)
                candidates.append((sgRNA, orig_pos, "-"))
    
    return candidates

def filter_sgrnas(sgrnas: List[Tuple[str, int, str]]) -> List[Tuple[str, int, str]]:
    """
    Filter sgRNAs based on design rules (GC content, homopolymers, etc.).
    """
    filtered = []
    for sgRNA, pos, strand in sgrnas:
        if valid_gc(sgRNA) and no_homopolymers(sgRNA) and no_self_complementarity(sgRNA):
            filtered.append((sgRNA, pos, strand))
    return filtered

def valid_gc(sgRNA: str, min_gc: float = 0.4, max_gc: float = 0.6) -> bool:
    """
    Check if GC content is within 40-60%.
    """
    gc = sum(1 for base in sgRNA if base in ["G", "C"]) / len(sgRNA)
    return min_gc <= gc <= max_gc

def no_homopolymers(sgRNA: str, max_repeats: int = 3) -> bool: 
    """
    Check for homopolymers (e.g., 'AAAA').
    """
    return not any(re.search(base * (max_repeats + 1), sgRNA) for base in "ATCG")

# no_self_complementarity: to avoid hairpins
def no_self_complementarity(sgRNA: str, seed_length: int = 8) -> bool:
    """
    Check for palindromic sequences in the seed region (last 8bp).
    """
    seed = sgRNA[-seed_length:]
    return str(Seq(seed).reverse_complement()) not in sgRNA[:-seed_length]