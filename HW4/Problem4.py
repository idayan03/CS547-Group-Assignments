import sys
import numpy as np
import matplotlib.pyplot as plt

def phi(n, x):
    """ Evaluates the function phi_n(x) = cos(2^n * x)

    Args:
        n : exponent of 2^n
        x : input value

    Returns:
        The result of the function phi_n(x) evaluated on x
    """
    return np.cos(np.power(2, n) * x)

def f_five(m_1, m_2, m_3):
    """ Evaluates the function f_5(m_1, m_2, m_3) = exp[pi*phi_3(m_3*phi_2(m_2*phi_1(m_1*x)))]
        
    x = 5 in the given homework problem

    Args:
        m_1 : scalar value of m_1 
        m_2 : scalar value of m_2
        m_3 : scalar value of m_3
        
    Returns:
        The result of the function f_5 evaluated on m_1, m_2, and m_3
    """
    first_inner_result = m_2 * phi(1, m_1 * 5)
    second_inner_result = m_3 * phi(2, first_inner_result)
    third_inner_result = np.pi * phi(3, second_inner_result)
    outer_result = np.exp(third_inner_result)
    return outer_result

def main():
    # Evaluating lim_{e->0} {f_5(10, 9, 8+e) - f_5{10, 9, 8}} / e
    epsilon = 1e-1
    m_3_epsilons = [epsilon]
    m_3_limits = [sys.maxsize, (f_five(10, 9, 8 + epsilon) - f_five(10, 9, 8)) / epsilon]
    while True:
        last_evaluated_limit = m_3_limits[len(m_3_limits) - 1]
        previous_evaluated_limit = m_3_limits[len(m_3_limits) - 2]
        if np.abs(last_evaluated_limit - previous_evaluated_limit) < 1e-5:
            break
        epsilon = epsilon / 2
        new_limit = (f_five(10, 9, 8 + epsilon) - f_five(10, 9, 8)) / epsilon
        m_3_limits.append(new_limit)
        m_3_epsilons.append(epsilon)
    
    # Evaluating lim_{e->0} {f_5(10, 9+e, 8) - f_5{10, 9, 8}} / e
    epsilon = 1e-1
    m_2_epsilons = [epsilon]
    m_2_limits = [sys.maxsize, (f_five(10, 9 + epsilon, 8) - f_five(10, 9, 8)) / epsilon]
    while True:
        last_evaluated_limit = m_2_limits[len(m_2_limits) - 1]
        previous_evaluated_limit = m_2_limits[len(m_2_limits) - 2]
        if np.abs(last_evaluated_limit - previous_evaluated_limit) < 1e-5:
            break
        epsilon = epsilon / 2
        new_limit = (f_five(10, 9 + epsilon, 8) - f_five(10, 9, 8)) / epsilon
        m_2_limits.append(new_limit)
        m_2_epsilons.append(epsilon)

    # Evaluating lim_{e->0} {f_5(10+e, 9, 8) - f_5{10, 9, 8}} / e
    epsilon = 1e-1
    m_1_epsilons = [epsilon]
    m_1_limits = [sys.maxsize, (f_five(10 + epsilon, 9, 8) - f_five(10, 9, 8)) / epsilon]
    while True:
        last_evaluated_limit = m_1_limits[len(m_1_limits) - 1]
        previous_evaluated_limit = m_1_limits[len(m_1_limits) - 2]
        if np.abs(last_evaluated_limit - previous_evaluated_limit) < 1e-5:
            break
        epsilon = epsilon / 2
        new_limit = (f_five(10 + epsilon, 9, 8) - f_five(10, 9, 8)) / epsilon
        m_1_limits.append(new_limit)
        m_1_epsilons.append(epsilon)

    plt.plot(m_3_epsilons, m_3_limits[1:], label="limit w.r.t m_3")
    plt.plot(m_2_epsilons, m_2_limits[1:], label="limit w.r.t m_2")
    plt.plot(m_1_epsilons, m_1_limits[1:], label="limit w.r.t m_1")
    plt.xlabel('epsilon')
    plt.ylabel('limit')
    plt.legend()
    plt.gca().invert_xaxis()
    plt.show()

if __name__ == "__main__":
    main()