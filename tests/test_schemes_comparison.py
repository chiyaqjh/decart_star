# decart/tests/test_schemes_comparison.py
import time

def compare_decart_and_decart_star():
    
    # Test parameters
    N, n = 128, 16
    users = [5, 6, 7, 8, 9, 10]
    
    # Test DeCart (O(n²))
    from schemes.decart import DeCartSystem, DeCartParams
    
    decart = DeCartSystem(DeCartParams(N=N, n=n))
    start = time.time()
    crs1, pp1, aux1 = decart.setup()
    setup_time1 = time.time() - start
    
    # Count parameters
    h_count1 = len(crs1['h_i'])
    H_count1 = len(crs1['H_ij'])
    
    # Test DeCart* (O(n))
    from schemes.decart_star import DeCartStarSystem, DeCartStarParams
    
    decart_star = DeCartStarSystem(DeCartStarParams(N=N, n=n))
    start = time.time()
    crs2, pp2, aux2 = decart_star.setup()
    setup_time2 = time.time() - start
    
    h_count2 = len([h for h in crs2['h_i'] if h])
    
    # Comparison results
    
    print(f"     Parameter comparison:")
    print(f"     DeCart  : h_i={h_count1}, H_ij={H_count1}, total={h_count1 + H_count1}")
    print(f"     DeCart* : h_i={h_count2}, total={h_count2}")
    print(f"     Optimization ratio: {(h_count1 + H_count1) / h_count2:.1f}:1")
    
    print(f"     Initialization time:")
    print(f"     DeCart  : {setup_time1*1000:.2f} ms")
    print(f"     DeCart* : {setup_time2*1000:.2f} ms")
    print(f"     Speedup: {setup_time1/setup_time2:.1f}x")
    
if __name__ == "__main__":
    compare_decart_and_decart_star()