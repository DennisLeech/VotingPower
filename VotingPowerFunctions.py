import numpy as np
import pandas as pd
from typing import List
from itertools import combinations
from decimal import Decimal, getcontext
from scipy import integrate, stats
from scipy.stats import norm
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

def validateipmmle(quota: Union[int, float], n1: int, n2: int, weights: List[Union[int, float]]) -> Tuple[float, int, int, np.ndarray]:
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
    quota, n1, n2, weights = validateipmmle(quota, n1, n2, weights)
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
    

def validateSSDirect(quota: Union[int, float], weights: List[Union[int, float]]) -> Tuple[float, np.ndarray]:
    """
    Validate inputs for the Shapley-Shubik index calculation.
    
    Args:
        quota: The quota needed to win
        weights: List of weights (votes) for each player
        
    Returns:
        Tuple of validated (quota, weights)
        
    Raises:
        ValueError: If inputs are invalid
    """
    # Type checking
    if not isinstance(quota, (int, float)):
        raise ValueError("Quota must be a number")
        
    # Convert weights to numpy array
    try:
        weights_array = np.array(weights, dtype=float)
    except (ValueError, TypeError):
        raise ValueError("All weights must be convertible to floating point numbers")
        
    # Value validation
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
        
    return float(quota), weights_array

def generate_subsets(n: int) -> np.ndarray:
    """Generate all possible subsets for n players."""
    if n > 30:  # 2^30 is about 1 billion combinations
        raise MemoryError("Too many players for exhaustive subset generation")
        
    return np.array([list(map(int, format(i, f'0{n}b'))) 
                    for i in range(2**n)])

def ssw(n: int, ns: int) -> float:
    """
    Calculate the Shapley-Shubik weighting factor.
    
    Args:
        n: Total number of players
        ns: Size of the coalition
        
    Returns:
        float: The weighting factor
    """
    if ns <= 0:
        return 0.0
    x = 1.0
    for j in range(1, ns):
        x = x * j / (n - j)
    return x / n

def ssdirect(
    quota: Union[int, float],
    weights: List[Union[int, float]],
    names: List[str] = None
) -> pd.DataFrame:
    """
    Calculate Shapley-Shubik power indices using direct enumeration.
    
    Args:
        quota: The quota needed to win
        weights: List of weights (votes) for each player
        names: Optional list of player names
        
    Returns:
        pandas.DataFrame containing weights and Shapley-Shubik indices
    """
    # Validate inputs
    quota, weights = validateSSDirect(quota, weights)
    n = len(weights)
    
    if names is None:
        names = [f"Player_{i+1}" for i in range(n)]
    elif len(names) != n:
        raise ValueError(f"Number of names ({len(names)}) must match number of weights ({n})")
    
    # Initialize arrays
    indices = np.zeros(n)
    
    # Generate all possible coalitions
    subsets = generate_subsets(n)
    
    # Process each coalition
    for subset in subsets:
        ns = np.sum(subset)
        total = np.sum(subset * weights)
        
        if total >= quota:
            # Check each player's contribution
            for i in range(n):
                if subset[i] == 1:
                    # Check if player is critical (coalition would lose without them)
                    coalition_without_i = total - weights[i]
                    if coalition_without_i < quota:
                        indices[i] += ssw(n, ns)
    
    # Create DataFrame with results
    results = pd.DataFrame({
        'Name': names,
        'Weight': weights,
        'Shapley_Shubik_Index': indices
    })
    
    return results

from typing import List, Union
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy import integrate
from itertools import product
import math


def ssgenf(
    quota: Union[int, float],
    weights: List[Union[int, float]],
    names: List[str] = None
) -> pd.DataFrame:
    """
    Calculate the Shapley-Shubik power index for a weighted voting system.
    
    Parameters:
    -----------
    quota : int or float
        The minimum total weight required for a coalition to win
    weights : list of int or float
        The weights of each voter
    names : list of str, optional
        Names of the voters. If None, integers starting from 1 will be used
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing names, weights, and Shapley-Shubik indices
    """
    # Convert inputs to numpy arrays
    weights_array = np.array(weights, dtype=np.float64)
    N = len(weights_array)
    
    # Create default names if not provided
    if names is None:
        names = [str(i+1) for i in range(N)]
    
    # Ensure names has the right length
    if len(names) != N:
        raise ValueError("Length of names must match length of weights")
    
    # Create arrays with correct sizing
    n = N + 1
    WW = np.zeros(2 * n, dtype=np.float64)
    WW[1:N+1] = weights_array  # Store weights in array (1-indexed as in original)
    
    # Calculate total weight
    W = int(np.sum(weights_array))
    
    # Initialize values
    Q = quota
    FF = np.zeros(2 * n, dtype=np.float64)
    
    # Initial setup
    I = 1
    W1 = W - WW[I]
    Q1 = Q - WW[I]
    
    # Create 2D array with NumPy
    C = np.zeros((W + 2, n + 1), dtype=np.float64)
    C[1, 1] = 1.0
    
    Y = (N + 1) // 2
    M = Y
    
    # Main calculation loops
    for R in range(2, N + 1):
        for K in range(int(M + 1), 1, -1):
            T = K - 1
            if WW[R] < T:
                for J in range(T + 1, int(Q) + 1):
                    C[J, K] += C[J - int(WW[R]), K - 1]
            else:
                T = int(WW[R])
                for J in range(T + 1, int(Q) + 1):
                    C[J, K] += C[J - T, K - 1]
    
    # Calculate first index
    F = 0.0
    A = 100.0 / N
    
    for K in range(0, N):
        K1 = N - K - 1
        k1 = K1
        T = K
        S = 0.0
        
        if Q1 > T:
            T = int(Q1)
        
        for J in range(T, int(Q)):
            if K > M:
                if J > W1:
                    F += A * S
                    if K1 != 0:
                        A = A * (K + 1) / k1
                        break
                else:
                    X = int(W1 - J + 1)
                    B = C[X, K1 + 1]
            else:
                B = C[J + 1, K + 1]
            S += B
        
        F += A * S
        if K1 != 0:
            A = A * (K + 1) / k1
    
    FF[1] = F
    
    # Calculate indices for remaining members
    for I in range(2, N + 1):
        if WW[I] != WW[I - 1]:
            R = I
            for K in range(2, int(M + 2)):
                T = K - 1
                if WW[R] >= T:
                    T = int(WW[R])
                for J in range(T + 1, int(Q) + 1):
                    C[J, K] -= C[J - int(WW[R]), K - 1]
            
            R = I - 1
            for K in range(int(M + 1), 1, -1):
                T = K - 1
                if WW[R] >= T:
                    T = int(WW[R])
                for J in range(T + 1, int(Q) + 1):
                    C[J, K] += C[J - int(WW[R]), K - 1]
            
            F = 0.0
            A = 100.0 / N
            W1 = W - WW[I]
            Q1 = Q - WW[I]
            
            for K in range(0, N):
                K1 = N - K - 1
                T = K
                S = 0.0
                
                if Q1 > T:
                    T = int(Q1)
                
                for J in range(T, int(Q)):
                    if K > M:
                        if J > W1:
                            F += A * S
                            if K1 != 0:
                                A = A * (K + 1) / K1
                                break
                        else:
                            X = int(W1 - J + 1)
                            B = C[X, K1 + 1]
                    else:
                        B = C[J + 1, K + 1]
                    S += B
                
                F += A * S
                if K1 != 0:
                    A = A * (K + 1) / K1
        FF[I] = F
    
    # Prepare results
    indices = [FF[i]/100 for i in range(1, N+1)]
    
    # Create DataFrame
    results = pd.DataFrame({
        'Name': names,
        'Weight': weights,
        'Shapley_Shubik_Index': indices
    })
    
    return results


def ssmmle(quota, n1, n2, weights):
    n = n1 + n2
    weights = np.array(weights[:n])
    epsr = 1.0e-5
    g = np.zeros(n)
    b1 = np.zeros(n)
    
    sumw1 = np.sum(weights[:n1])
    sumw2 = np.sum(weights[n1:])
    ss2 = np.sum(weights[n1:]**2)
    sumw = sumw1 + sumw2
    
    js = [0] * n1
    more = False
    j = 0
    ncard = 0
    
    while True:
        if not more:
            m = 0
            more = True
            js = [0] * n1
            ncard = 0
        else:
            m = m + 1
            m1 = m
            j = 1
            
            while m1 % 2 != 1:
                j = j + 1
                m1 = m1 // 2
                
            l = js[j-1]
            js[j-1] = 1 - l
            ncard = ncard + 1 - 2*l
            
            if ncard != 1 or js[n1-1] == 0:
                more = True
            else:
                more = False
        
        ns = sum(js)
        a = sum(weights[:n1] * np.array(js))
        
        z = sumw2
        v = ss2
        
        if a < quota:
            for i in range(n1):
                if js[i] == 0:
                    lors = 1
                    a2 = quota - a
                    a1 = a2 - weights[i]
                    
                    sig = math.sqrt(0.5 * 0.5 * v)
                    if sig > 0:
                        c1 = (a1 - 0.5 * z) / sig
                        c2 = (a2 - 0.5 * z) / sig
                        sig_val = (0.5 ** ns) * ((0.5) ** (n1 - ns - lors))
                        a3 = (stats.norm.cdf(c2) - stats.norm.cdf(c1)) * sig_val
                        b1[i] += a3
                    
                    if sumw2 >= quota - a - weights[i]:
                        n_points = 100
                        h = 1.0 / n_points
                        integral = 0
                        
                        for k in range(n_points + 1):
                            x = k * h
                            if k == 0 or k == n_points:
                                weight = 0.5
                            else:
                                weight = 1.0
                                
                            sig = math.sqrt(x * (1.0 - x) * v)
                            if sig > 0:
                                c1 = (a1 - x * z) / sig
                                c2 = (a2 - x * z) / sig
                                sig_val = (x ** ns) * ((1.0 - x) ** (n1 - ns - lors))
                                f_val = (stats.norm.cdf(c2) - stats.norm.cdf(c1)) * sig_val
                                integral += weight * f_val
                        
                        g[i] += h * integral
        
        if sumw2 >= quota - a:
            for i in range(n1, n):
                z_i = sumw2 - weights[i]
                v_i = ss2 - weights[i]**2
                a2 = quota - a
                a1 = a2 - weights[i]
                lors = 0
                
                sig = math.sqrt(0.5 * 0.5 * v_i)
                if sig > 0:
                    c1 = (a1 - 0.5 * z_i) / sig
                    c2 = (a2 - 0.5 * z_i) / sig
                    sig_val = (0.5 ** ns) * ((0.5) ** (n1 - ns - lors))
                    a3 = (stats.norm.cdf(c2) - stats.norm.cdf(c1)) * sig_val
                    b1[i] += a3
                
                n_points = 100
                h = 1.0 / n_points
                integral = 0
                
                for k in range(n_points + 1):
                    x = k * h
                    if k == 0 or k == n_points:
                        weight = 0.5
                    else:
                        weight = 1.0
                        
                    sig = math.sqrt(x * (1.0 - x) * v_i)
                    if sig > 0:
                        c1 = (a1 - x * z_i) / sig
                        c2 = (a2 - x * z_i) / sig
                        sig_val = (x ** ns) * ((1.0 - x) ** (n1 - ns - lors))
                        f_val = (stats.norm.cdf(c2) - stats.norm.cdf(c1)) * sig_val
                        integral += weight * f_val
                
                g[i] += h * integral
        
        if not more:
            break
    
    sumb1 = np.sum(b1)
    if sumb1 > 0:
        b = b1 / sumb1
    else:
        b = np.zeros(n)
    
    sumg = np.sum(g)
    if sumg > 0:
        g = g / sumg
    
    results = pd.DataFrame({
        'Weight': weights,
        'Shapley_Shubik_Index': g
    })
        
    return results
    #return g.tolist()



def ssocean(Q, N1, ABSW, SUMW):
    """
    Computes Shapley-Shubik indices for an "oceanic" game.
    Direct translation from Fortran to Python with NumPy.
    
    Args:
        Q: Quota
        N1: Number of "atomic" players
        ABSW: Weights of atomic players
        SUMW: Total sum of weights
    
    Returns:
        DataFrame containing player weights and SS indices
    """
    # Initialize variables
    NN = 1000
    EPSR = 1.0e-3
    
    W = np.zeros(NN)
    ABSW = np.array(ABSW)
    G = np.zeros(NN)
    
    Q = Q / SUMW
    SUMW1 = 0.0
    
    for I in range(N1):
        W[I] = ABSW[I] / SUMW
        SUMW1 += W[I]
        G[I] = 0.0
    
    SUMW2 = 1 - SUMW1
    
    # Define BRN function from the original code
    def BRN(X):
        if X < 0.0:
            return 0.0
        elif X > 1.0:
            return 1.0
        else:
            return X
    
    # Generate all possible subsets of N1 players
    for subset in product([0, 1], repeat=N1):
        IS = np.array(subset)
        
        NS = np.sum(IS)
        A = np.sum(IS * W[:N1])
        
        if A < Q:
            for I in range(N1):
                if IS[I] == 0:
                    if SUMW2 > Q - A - W[I]:
                        T1 = (Q - A - W[I]) / SUMW2
                        T2 = (Q - A) / SUMW2
                        T1 = BRN(T1)
                        T2 = BRN(T2)
                        
                        # Define integration function for this particular subset
                        def FUN(X):
                            return (X**NS) * ((1.0 - X)**(N1 - NS - 1))
                        
                        # Replace D01AHF with scipy's integrate.quad
                        result, _ = integrate.quad(FUN, T1, T2, epsabs=EPSR)
                        G[I] += result
    
    # Calculate final results
    SUMW2_original = SUMW2 * SUMW
    SUMG1 = sum(G[:N1])
    G2 = 1 - SUMG1
    
    # Create pandas DataFrame
    data = {
        'Player': ['Player ' + str(i+1) for i in range(N1)] + ['Non-atomic players'],
        'Weight': np.append(ABSW[:N1], SUMW2_original),
        'SS-Index': np.append(G[:N1], G2)
    }
    
    df = pd.DataFrame(data)
    
    return df

