"""
Model Checker - Compares discovered SR model to the true kinetic model.
Checks that the discovered model has coefficients PRESENT where expected.

The actual coefficient VALUES don't matter - only that they exist (are not 1).

Example TRUE model: (-7*A + 3*B) / (4*A + 2*B + 6)

  PASS: (-6*A + 2*B) / (5*A + 8*B + 3)  
        → Has coefficients on all terms (values differ, but that's OK)
        
  FAIL: (A + B) / (A + 2*B + 6)
        → Missing coefficients on A and B in numerator, and A in denominator
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


def extract_coefficients(poly_expr, variables):
    """
    Extract the coefficients of each monomial term in a polynomial.
    
    Returns a dict: {(power_A, power_B): coefficient_value}
    E.g., 4*A + 2*B + 6 -> {(1,0): 4, (0,1): 2, (0,0): 6}
    """
    try:
        expanded = expand(poly_expr)
        
        if expanded.is_number:
            return {(0, 0): float(expanded)}
        
        poly = Poly(expanded, *variables)
        coeffs = {}
        for monom, coeff in zip(poly.monoms(), poly.coeffs()):
            coeffs[monom] = float(coeff)
        
        return coeffs
    except Exception as e:
        return None


def coefficients_present_match(struct1, struct2, unity_tolerance=0.1):
    """
    Check if coefficients are PRESENT where expected.
    
    A coefficient is "present" if its absolute value is NOT close to 1.0.
    This checks that the discovered model has explicit coefficients (not just 1)
    on the same terms where the true model has coefficients.
    
    Args:
        struct1: Discovered model structure
        struct2: True model structure  
        unity_tolerance: How close to 1.0 to be considered "no coefficient" (default 0.1)
    
    Returns:
        (match: bool, details: str)
    """
    if struct1 is None or struct2 is None:
        return False, "Structure extraction failed"
    
    def has_explicit_coefficient(coeff_value, tolerance=0.1):
        """Check if a coefficient is explicitly present (not 1 or -1)."""
        return abs(abs(coeff_value) - 1.0) > tolerance
    
    def compare_coefficient_presence(expr1, expr2, part_name):
        coeffs1 = extract_coefficients(expr1, [A, B])
        coeffs2 = extract_coefficients(expr2, [A, B])
        
        if coeffs1 is None or coeffs2 is None:
            return False, f"Could not extract coefficients from {part_name}"
        
        # Check same monomials (structural match)
        if set(coeffs1.keys()) != set(coeffs2.keys()):
            return False, f"Different terms in {part_name}"
        
        # For each term, check if coefficient presence matches
        missing_coeffs = []
        for monom in coeffs2:
            true_coeff = coeffs2[monom]
            disc_coeff = coeffs1[monom]
            
            true_has_coeff = has_explicit_coefficient(true_coeff)
            disc_has_coeff = has_explicit_coefficient(disc_coeff)
            
            # If true model has a coefficient, discovered must also have one
            if true_has_coeff and not disc_has_coeff:
                term_name = _monom_to_str(monom)
                missing_coeffs.append(f"{term_name} (discovered={disc_coeff:.2f}, expected coeff like {true_coeff:.2f})")
        
        if missing_coeffs:
            return False, f"Missing coefficients in {part_name}: {', '.join(missing_coeffs)}"
        
        return True, f"{part_name} has all required coefficients"
    
    # Compare numerator coefficient presence
    num_match, num_detail = compare_coefficient_presence(
        struct1['numer_expr'], struct2['numer_expr'], "numerator"
    )
    if not num_match:
        return False, num_detail
    
    # Compare denominator coefficient presence
    den_match, den_detail = compare_coefficient_presence(
        struct1['denom_expr'], struct2['denom_expr'], "denominator"
    )
    if not den_match:
        return False, den_detail
    
    return True, "All required coefficients are present"


def _monom_to_str(monom):
    """Convert a monomial tuple to a string representation."""
    a_pow, b_pow = monom
    if a_pow == 0 and b_pow == 0:
        return "const"
    elif a_pow == 1 and b_pow == 0:
        return "A"
    elif a_pow == 0 and b_pow == 1:
        return "B"
    elif a_pow == 2 and b_pow == 0:
        return "A²"
    elif a_pow == 0 and b_pow == 2:
        return "B²"
    elif a_pow == 1 and b_pow == 1:
        return "A*B"
    else:
        return f"A^{a_pow}*B^{b_pow}"


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


def check_exact_match(discovered_model, true_model, tolerance=0.05):
    """
    Check if discovered model has the same structure AND has coefficients present
    where the true model has them.
    
    Args:
        discovered_model: String representation of the discovered equation
        true_model: String representation of the true equation
        tolerance: Not used for value comparison, kept for API compatibility
    
    Returns: (is_match, is_structural_match, similarity_score, details)
    """
    # Parse models
    discovered_expr = parse_model_to_sympy(discovered_model)
    true_expr = parse_model_to_sympy(true_model)
    
    if discovered_expr is None or true_expr is None:
        return False, False, 0.0, {"error": "Parse error"}
    
    # Extract structures
    disc_struct = extract_structure(discovered_expr)
    true_struct = extract_structure(true_expr)
    
    disc_desc = describe_structure(disc_struct)
    true_desc = describe_structure(true_struct)
    
    # Check structural match first
    is_structural_match = structures_match(disc_struct, true_struct)
    
    # Check coefficient presence (only if structural match passes)
    is_coeff_present = False
    coeff_detail = "Structure mismatch - coefficients not checked"
    if is_structural_match:
        is_coeff_present, coeff_detail = coefficients_present_match(disc_struct, true_struct)
    
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
    
    details = {
        'disc_structure': disc_desc,
        'true_structure': true_desc,
        'coeff_detail': coeff_detail
    }
    
    # Match requires both structural match and coefficient presence
    is_match = is_structural_match and is_coeff_present
    
    return is_match, is_structural_match, similarity, details


def check_discovered_model(best_model_equation, tolerance=0.05):
    """
    Check if the discovered model has correct structure AND coefficients present
    where expected (values don't need to match, just presence of non-unity coefficients).
    
    Args:
        best_model_equation: String representation of the rate equation for species A
        tolerance: Not used, kept for API compatibility
        
    Returns:
        (is_exact_match, similarity_score)
    """
    true_model = config.TRUE_MODEL
    
    print(f"Model Check (Exact Match with Coefficients):")
    print(f"  Discovered: {best_model_equation}")
    print(f"  True:       {true_model}")
    print(f"  Tolerance:  {tolerance*100:.0f}%")
    
    is_exact, is_structural, similarity, details = check_exact_match(
        best_model_equation, true_model, tolerance
    )
    
    print(f"  Discovered structure: {details['disc_structure']}")
    print(f"  True structure:       {details['true_structure']}")
    print(f"  Numerical similarity: {similarity:.4f}")
    print(f"  STRUCTURAL Match: {'YES ✓' if is_structural else 'NO ✗'}")
    print(f"  Coefficient detail: {details['coeff_detail']}")
    print(f"  EXACT Match (with coefficients): {'YES ✓' if is_exact else 'NO ✗'}")
    
    return is_exact, similarity


if __name__ == "__main__":
    print("=" * 60)
    print("Testing coefficient PRESENCE matching")
    print("(structure must match AND coefficients must be present where expected)")
    print("=" * 60)
    
    # Test 1: Exact same coefficients - should PASS
    print("\n--- Test 1: Exact same coefficients (should PASS) ---")
    test_model = "(-7*A + 3*B) / (4*A + 2*B + 6)"
    is_match, score = check_discovered_model(test_model)
    print(f"Result: Match={is_match}")
    
    # Test 2: Different coefficient VALUES but all present - should PASS
    print("\n--- Test 2: Different values but coefficients present (should PASS) ---")
    test_model_2 = "(-6*A + 2*B) / (5*A + 8*B + 3)"
    is_match_2, score_2 = check_discovered_model(test_model_2)
    print(f"Result: Match={is_match_2}")
    
    # Test 3: Missing coefficients (values are 1) - should FAIL
    print("\n--- Test 3: Missing coefficients on A,B in numerator (should FAIL) ---")
    test_model_3 = "(A + B) / (4*A + 2*B + 6)"
    is_match_3, score_3 = check_discovered_model(test_model_3)
    print(f"Result: Match={is_match_3}")
    
    # Test 4: Missing coefficient only on A in denominator - should FAIL
    print("\n--- Test 4: Missing coefficient on A in denominator (should FAIL) ---")
    test_model_4 = "(-7*A + 3*B) / (A + 2*B + 6)"
    is_match_4, score_4 = check_discovered_model(test_model_4)
    print(f"Result: Match={is_match_4}")
    
    # Test 5: Different structure - should FAIL
    print("\n--- Test 5: Different structure (should FAIL) ---")
    test_model_5 = "(-7*A + 3*B) / (4*A + 2*B)"
    is_match_5, score_5 = check_discovered_model(test_model_5)
    print(f"Result: Match={is_match_5}")


