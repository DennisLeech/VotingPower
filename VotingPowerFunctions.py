import numpy as np
import pandas as pd
from typing import List
from itertools import combinations
from decimal import Decimal, getcontext

def ipdirect(quota: int, weights: List[int]) -> pd.DataFrame:
    """
    Calculate power indices exactly by searching over all possible coalitions
    
    Args:
        quota: int, the quota needed for a winning coalition
        weights: list of int, the voting weights for each player
    
    Returns:
        pd.DataFrame: DataFrame containing all power indices for each player
    """
    n = len(weights)
    w = np.array(weights)
    
    # Initialize arrays
    ix = np.zeros(n, dtype=int)  # swing counts
    ncoals = 2**(n-1)  # number of coalitions per player
    ia = 0  # count of winning coalitions
    
    # Generate all possible coalitions using binary numbers
    for i in range(2**n):
        # Convert number to binary array of coalition membership
        coalition = np.array([int(x) for x in format(i, f'0{n}b')])
        
        # Calculate total weight of coalition
        tot = np.sum(coalition * w)
        
        # Check if it's a winning coalition
        if tot >= quota:
            ia += 1
            # Check each player to see if they're critical
            for j in range(n):
                if coalition[j] == 1:  # only check if player is in coalition
                    # Remove player and check if coalition becomes losing
                    if tot - w[j] < quota:
                        ix[j] += 1

    # Calculate indices
    b1 = ix / ncoals  # Non-normalized Banzhaf
    sumb1 = np.sum(b1)
    pta = 0.5 * ia / ncoals  # Power to Act
    
    # Calculate other indices
    b = b1 / sumb1  # Normalized Banzhaf
    ppa = b1 / (pta * 2) if pta != 0 else np.zeros_like(b1)  # PPA index
    pia = np.where(pta == 1, b1, b1 / ((1 - pta) * 2))  # PIA index
    
    # Create results dictionary
    results = []
    for i in range(n):
        results.append({
            'Player': i + 1,
            'Weight': weights[i],
            'Banzhaf_Index': float(b[i]),
            'Normalized_Banzhaf': float(b1[i]),
            'PPA': float(ppa[i]),
            'PIA': float(pia[i]),
            'PTA': float(pta)
        })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Print results
    print("\nPower Indices Results:")
    print(f"Number of winning coalitions: {ia}")
    print(df.to_string(index=False))
    
    return df
    


# Set high precision for decimal calculations
getcontext().prec = 400

def ipgenf(quota: int, weights: list) -> pd.DataFrame:
    """
    Calculate various power indices for weighted voting systems
    
    Args:
        quota: int, the quota needed for a winning coalition
        weights: list of int, the voting weights for each player
    
    Returns:
        pd.DataFrame: DataFrame containing all power indices for each player
    """
    N = len(weights)  # number of players
    W = np.array([0] + weights, dtype=int)  # add 0 at index 0 to match C++ indexing
    WW = int(np.sum(weights))  # sum of weights
    
    # Initialize A array (corresponds to A[] in C++)
    A = np.zeros(WW + 1, dtype=object)
    A[0] = 1
    
    # Calculate A array (dynamic programming for coalition counting)
    SR = 0
    for R in range(1, N + 1):
        SR = int(SR + W[R])
        for J in range(int(SR), 0, -1):
            if J >= W[R]:
                A[J] = A[J] + A[J - W[R]]
    
    # Calculate WIN (number of winning coalitions)
    WIN = sum(A[quota:])
    
    # Calculate ETA (swing votes) for each player
    ETA = np.zeros(N + 1, dtype=object)
    SUMETA = 0
    
    for i in range(1, N + 1):
        # Initialize C array
        C = np.zeros(WW + 1, dtype=object)
        C[0] = 1
        
        # Calculate C array
        for J in range(1, WW - W[i] + 1):
            if J < W[i]:
                C[J] = A[J]
            else:
                C[J] = A[J] - C[J - W[i]]
        
        # Calculate ETA[i]
        lower_bound = max(0, quota - W[i])
        ETA[i] = sum(C[lower_bound:quota])
        SUMETA += ETA[i]
    
    # Calculate power indices
    results = []
    for i in range(1, N + 1):
        # Convert to Decimal for high precision
        eta_i = Decimal(str(ETA[i]))
        sum_eta = Decimal(str(SUMETA))
        win_d = Decimal(str(WIN))
        two_n = Decimal(2) ** N
        two_n_minus_1 = Decimal(2) ** (N-1)
        
        # Calculate indices
        BZ = eta_i / sum_eta  # Banzhaf index
        BZNN = eta_i / two_n_minus_1  # Normalized Banzhaf index
        PPA = eta_i / win_d  # PPA index
        
        # Calculate PIA (Coleman's Initiative Power)
        if WIN == two_n:
            PIA = eta_i
        else:
            PIA = eta_i / (two_n - win_d)
        
        # Calculate PTA (Coleman's Preventive Power)
        PTA = eta_i / two_n
        
        results.append({
            'Player': i,
            'Weight': int(W[i]),
            'Swings': int(ETA[i]),
            'Banzhaf_Index': float(BZ),
            'Normalized_Banzhaf': float(BZNN),
            'PPA': float(PPA),
            'PIA': float(PIA),
            'PTA': float(PTA)
        })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Print the DataFrame
    print("\nPower Indices Results:")
    print(df.to_string(index=False))
    
    return df
