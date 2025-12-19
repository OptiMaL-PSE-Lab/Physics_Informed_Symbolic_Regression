"""
Model Checker - Compares discovered SR model to the true kinetic model.
Checks for STRUCTURAL match (algebraic form), not exact coefficient values.

Example: (-7*A + 3*B) / (4*A + 2*B + 6) should match
         (-1.5*A + 0.8*B) / (2*A + 1*B + 3) because they have the same structure.
"""

import numpy as np
import sympy as sp
from sympy import symbols, simplify, expand, Poly, fraction, degree, Add, Mul, Symbol
from sympy.core.numbers import Float, Integer, Rational, Number
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

A, B = symbols('A B')


def parse_model_to_sympy(model_str):
    """Convert a model string to a sympy expression."""
    model_str = str(model_str).replace("**", "^")
    try:
        expr = sp.sympify(model_str, locals={'A': A, 'B': B})
        return expr
    except Exception as e:
        print(f"  Parse error: {e}")
        return None


def get_polynomial_structure(poly_expr, variables):
    """
    Get the structural form of a polynomial expression.
    Returns a set of monomial patterns (tuples of variable powers).
    
    E.g., 4*A + 2*B + 6 -> {(1,0), (0,1), (0,0)} meaning A, B, constant
    """
    try:
        # Expand to get polynomial form
        expanded = expand(poly_expr)
        
        # If it's just a number, return constant structure
        if expanded.is_number:
            return frozenset({(0, 0)})
        
        # Get as polynomial
        poly = Poly(expanded, *variables)
        monomials = poly.monoms()
        
        return frozenset(monomials)
    except Exception as e:
        # Fallback: try to identify terms manually
        return None


def extract_structure(expr):
    """
    Extract the algebraic structure of an expression.
    
    Returns a structure descriptor that can be compared between expressions.
    The structure ignores coefficient values but captures:
    - Whether it's a ratio (numerator/denominator)
    - What monomial terms appear in numerator and denominator
    """
    if expr is None:
        return None
    
    try:
        # Expand and simplify
        expr = simplify(expr)
        
        # Get numerator and denominator
        numer, denom = fraction(expr)
        
        # Get polynomial structure of each
        numer_struct = get_polynomial_structure(numer, [A, B])
        denom_struct = get_polynomial_structure(denom, [A, B])
        
        return {
            'is_ratio': not denom.is_number or denom != 1,
            'numerator': numer_struct,
            'denominator': denom_struct,
            'numer_expr': numer,
            'denom_expr': denom
        }
    except Exception as e:
        print(f"  Structure extraction error: {e}")
        return None


def structures_match(struct1, struct2):
    """
    Check if two structures are equivalent.
    """
    if struct1 is None or struct2 is None:
        return False
    
    # Both must be ratios or both not
    if struct1['is_ratio'] != struct2['is_ratio']:
        return False
    
    # Compare monomial structures
    if struct1['numerator'] != struct2['numerator']:
        return False
    
    if struct1['denominator'] != struct2['denominator']:
        return False
    
    return True


def describe_structure(struct):
    """Human-readable description of a structure."""
    if struct is None:
        return "Unknown"
    
    def describe_monomials(monoms):
        if monoms is None:
            return "complex"
        terms = []
        for powers in sorted(monoms):
            a_pow, b_pow = powers
            if a_pow == 0 and b_pow == 0:
                terms.append("const")
            elif a_pow == 1 and b_pow == 0:
                terms.append("A")
            elif a_pow == 0 and b_pow == 1:
                terms.append("B")
            elif a_pow == 2 and b_pow == 0:
                terms.append("A²")
            elif a_pow == 0 and b_pow == 2:
                terms.append("B²")
            elif a_pow == 1 and b_pow == 1:
                terms.append("A*B")
            else:
                terms.append(f"A^{a_pow}*B^{b_pow}")
        return " + ".join(terms) if terms else "0"
    
    numer_desc = describe_monomials(struct['numerator'])
    denom_desc = describe_monomials(struct['denominator'])
    
    if struct['is_ratio']:
        return f"({numer_desc}) / ({denom_desc})"
    else:
        return numer_desc


def check_structural_match(discovered_model, true_model):
    """
    Check if discovered model has the same STRUCTURE as the true model.
    Coefficients can differ, only the algebraic form must match.
    
    Returns: (is_structural_match, similarity_score, structure_descriptions)
    """
    # Parse models
    discovered_expr = parse_model_to_sympy(discovered_model)
    true_expr = parse_model_to_sympy(true_model)
    
    if discovered_expr is None or true_expr is None:
        return False, 0.0, ("Parse error", "Parse error")
    
    # Extract structures
    disc_struct = extract_structure(discovered_expr)
    true_struct = extract_structure(true_expr)
    
    disc_desc = describe_structure(disc_struct)
    true_desc = describe_structure(true_struct)
    
    # Check structural match
    is_match = structures_match(disc_struct, true_struct)
    
    # Calculate numerical similarity for reference
    try:
        test_A = np.linspace(1, 10, 30)
        test_B = np.linspace(0.1, 5, 30)
        
        discovered_func = sp.lambdify((A, B), discovered_expr, 'numpy')
        true_func = sp.lambdify((A, B), true_expr, 'numpy')
        
        total_error = 0
        total_true = 0
        count = 0
        
        for a in test_A:
            for b in test_B:
                try:
                    d_val = float(discovered_func(a, b))
                    t_val = float(true_func(a, b))
                    
                    if not (np.isnan(d_val) or np.isnan(t_val) or 
                            np.isinf(d_val) or np.isinf(t_val)):
                        total_error += (d_val - t_val) ** 2
                        total_true += t_val ** 2
                        count += 1
                except:
                    continue
        
        if count > 0 and total_true > 0:
            rmse = np.sqrt(total_error / count)
            true_scale = np.sqrt(total_true / count)
            relative_error = rmse / (true_scale + 1e-10)
            similarity = 1.0 / (1.0 + relative_error)
        else:
            similarity = 0.0
    except:
        similarity = 0.0
    
    return is_match, similarity, (disc_desc, true_desc)


def check_discovered_model(best_model_equation):
    """
    Main function to check if the best discovered model has the same
    STRUCTURAL FORM as the true model (coefficients don't need to match).
    
    Args:
        best_model_equation: String representation of the rate equation for species A
        
    Returns:
        (is_structural_match, similarity_score)
    """
    true_model = config.TRUE_MODEL
    
    print(f"Model Check (Structural Match):")
    print(f"  Discovered: {best_model_equation}")
    print(f"  True:       {true_model}")
    
    is_match, similarity, (disc_struct, true_struct) = check_structural_match(
        best_model_equation, true_model
    )
    
    print(f"  Discovered structure: {disc_struct}")
    print(f"  True structure:       {true_struct}")
    print(f"  Numerical similarity: {similarity:.4f}")
    print(f"  STRUCTURAL Match: {'YES ✓' if is_match else 'NO ✗'}")
    
    return is_match, similarity


if __name__ == "__main__":
    print("=" * 60)
    print("Testing STRUCTURAL match (coefficients can differ)")
    print("=" * 60)
    
    # Test 1: Exact same structure and coefficients
    print("\n--- Test 1: Exact same ---")
    test_model = "(-7*A + 3*B) / (4*A + 2*B + 6)"
    is_match, score = check_discovered_model(test_model)
    print(f"Result: Match={is_match}")
    
    # Test 2: Same STRUCTURE, different coefficients
    print("\n--- Test 2: Same structure, different coefficients ---")
    test_model_2 = "(-1.5*A + 0.8*B) / (2*A + 1*B + 3)"
    is_match_2, score_2 = check_discovered_model(test_model_2)
    print(f"Result: Match={is_match_2}")
    
    # Test 3: Different structure (no constant in denominator)
    print("\n--- Test 3: Different structure (missing constant) ---")
    test_model_3 = "(-7*A + 3*B) / (4*A + 2*B)"
    is_match_3, score_3 = check_discovered_model(test_model_3)
    print(f"Result: Match={is_match_3}")
    
    # Test 4: Different structure (no B in numerator)
    print("\n--- Test 4: Different structure (missing B term) ---")
    test_model_4 = "(-7*A) / (4*A + 2*B + 6)"
    is_match_4, score_4 = check_discovered_model(test_model_4)
    print(f"Result: Match={is_match_4}")
    
    # Test 5: Completely different structure
    print("\n--- Test 5: Completely different ---")
    test_model_5 = "A * B / (A + B)"
    is_match_5, score_5 = check_discovered_model(test_model_5)
    print(f"Result: Match={is_match_5}")
