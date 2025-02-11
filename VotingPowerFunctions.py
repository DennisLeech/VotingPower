import numpy as np
import pandas as pd
from typing import List
from itertools import combinations
from decimal import Decimal, getcontext
from scipy import stats
from scipy import integrate
from typing import List, Tuple, Union, Optional
import warnings


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
    
import numpy as np
import pandas as pd
from scipy.stats import norm


def validate_inputs(quota: Union[int, float], n1: int, n2: int, weights: List[Union[int, float]]) -> Tuple[float, int, int, np.ndarray]:
    """
    Validate all inputs for the voting power analysis.
    
    Args:
        quota: The quota needed to win
        n1: Number of players in large group
        n2: Number of players in small group
        weights: List of weights for all players
        
    Returns:
        Tuple of validated (quota, n1, n2, weights)
        
    Raises:
        ValueError: If any inputs are invalid
    """
    # Type checking
    if not isinstance(quota, (int, float)):
        raise ValueError("Quota must be a number")
    if not isinstance(n1, int):
        raise ValueError("n1 must be an integer")
    if not isinstance(n2, int):
        raise ValueError("n2 must be an integer")
        
    # Convert weights to numpy array
    try:
        weights_array = np.array(weights, dtype=float)
    except (ValueError, TypeError):
        raise ValueError("All weights must be convertible to floating point numbers")
        
    # Value validation
    if n1 < 0 or n2 < 0:
        raise ValueError("Number of players cannot be negative")
    if n1 + n2 != len(weights):
        raise ValueError(f"Sum of n1 ({n1}) and n2 ({n2}) must equal number of weights ({len(weights)})")
    if quota <= 0:
        raise ValueError("Quota must be positive")
    if any(w < 0 for w in weights):
        raise ValueError("All weights must be non-negative")
    
    # Game property validation
    total_weight = np.sum(weights_array)
    if quota > total_weight:
        raise ValueError(f"Quota ({quota}) cannot exceed total weight ({total_weight})")
    if quota <= total_weight / 2:
        warnings.warn("Quota is not a strict majority, which may lead to paradoxical results")
        
    return float(quota), n1, n2, weights_array

def generate_subsets(n1: int) -> np.ndarray:
    """Generate all possible subsets for n1 players."""
    if n1 > 30:  # 2^30 is about 1 billion combinations
        raise MemoryError("Too many players for exhaustive subset generation")
        
    return np.array([list(map(int, format(i, f'0{n1}b'))) 
                    for i in range(2**n1)])

def compute_f(x: float, a1: float, a2: float, z: float, v: float, ns: int, n1: int, lors: int) -> float:
    """Compute the F function used in power calculations."""
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            sig = np.sqrt(x * (1.0 - x) * v)
            if sig == 0:
                return 0.0
                
            c1 = (a1 - x * z) / sig
            c2 = (a2 - x * z) / sig
            
            prob_factor = (x**ns) * ((1.0 - x)**(n1 - ns - lors))
            return (norm.cdf(c2) - norm.cdf(c1)) * prob_factor
            
        except RuntimeWarning:
            warnings.warn(f"Numerical stability issue in compute_f: x={x}, v={v}, ns={ns}")
            return 0.0

def ipmmle(
    quota: Union[int, float],
    n1: int,
    n2: int,
    weights: List[Union[int, float]],
    names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Analyze voting power for a weighted voting game.
    
    Args:
        quota: The quota needed to win
        n1: Number of players in large group
        n2: Number of players in small group
        weights: List of weights for all players
        names: Optional list of player names
        
    Returns:
        pandas.DataFrame containing the analysis results
    """
    # Validate inputs
    quota, n1, n2, weights = validate_inputs(quota, n1, n2, weights)
    n = n1 + n2
    
    if names is None:
        names = [f"Player_{i+1}" for i in range(n)]
    elif len(names) != n:
        raise ValueError(f"Number of names ({len(names)}) must match number of players ({n})")
    
    # Calculate initial sums
    sum_w1 = np.sum(weights[:n1])
    sum_w2 = np.sum(weights[n1:])
    ss2 = np.sum(weights[n1:]**2)
    
    # Initialize results
    b1 = np.zeros(n)
    pta = 0.0
    
    # Generate and process all subsets
    subsets = generate_subsets(n1)
    
    for subset in subsets:
        ns = np.sum(subset)
        a = np.sum(subset * weights[:n1])
        
        if a >= quota:
            pta += 2**(-n1)
        else:
            z = sum_w2
            v = ss2
            a2 = sum_w2
            a1 = quota - a
            
            pta += compute_f(0.5, a1, a2, z, v, ns, n1, 0)
            
            # Calculate individual player contributions
            for i in range(n1):
                if subset[i] == 0:
                    a2 = quota - a
                    a1 = a2 - weights[i]
                    b1[i] += compute_f(0.5, a1, a2, z, v, ns, n1, 1)
            
            if sum_w2 >= quota - a:
                for i in range(n1, n):
                    z = sum_w2 - weights[i]
                    v = ss2 - weights[i]**2
                    a2 = quota - a
                    a1 = a2 - weights[i]
                    b1[i] += compute_f(0.5, a1, a2, z, v, ns, n1, 0)
    
    # Calculate indices
    sum_b1 = np.sum(b1)
    if sum_b1 == 0:
        warnings.warn("All raw scores are zero, results may not be meaningful")
        normalized_indices = np.zeros_like(b1)
    else:
        normalized_indices = b1 / sum_b1
    
    if pta == 0:
        warnings.warn("Power to act is zero, preventive power indices may not be meaningful")
        ppa = np.zeros_like(b1)
        pia = np.zeros_like(b1)
    else:
        ppa = b1 / (pta * 2)
        pia = b1 / ((1 - pta) * 2) if pta < 1 else np.zeros_like(b1)
    
    # Create DataFrame
    results = pd.DataFrame({
        'Name': names,
        'Weight': weights,
        'Raw_Banzhaf_Score': b1,
        'Normalized_Banzhaf_Index': normalized_indices,
        'Preventive_Power_Index': ppa,
        'Initiative_Power_Index': pia
    })
    
    # Add power to act as metadata
    results.attrs['power_to_act'] = pta
    
    return results
