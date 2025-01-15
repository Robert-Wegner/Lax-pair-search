from sympy import *
from IPython.display import display
from sympy.core.decorators import call_highest_priority
from sympy import Expr, Matrix, Mul, Add, diff, Function
from sympy.core.numbers import Zero
import traceback

def Equ(*args, **kwargs):
    kwargs['evaluate'] = False
    return Eq(*args, **kwargs)


class D(Expr):
    _op_priority = 11.
    is_commutative = False
    diff_symbols = []
    diff_symbols_nc = []
    non_diff_symbols = []
    non_diff_symbols_nc = []

    def __init__(self, *variables, **assumptions):
        super(D, self).__init__()
        self.evaluate = False
        self.variables = variables

    def __repr__(self):
        return 'D%s' % str(self.variables)

    def __str__(self):
        return self.__repr__()

    @call_highest_priority('__mul__')
    def __rmul__(self, other):
        return Mul(other, self)

    @call_highest_priority('__rmul__')
    def __mul__(self, other):
        if isinstance(other, D):
            variables = self.variables + other.variables
            return D(*variables)
        
        if isinstance(other, Matrix):
            other_copy = other.copy()
            for i, elem in enumerate(other):
                other_copy[i] = self * elem
            return other_copy

        if self.evaluate:
            return D.multi_deriv(other, *self.variables)
        else:
            return Mul(self, other)

    def __pow__(self, other):
        variables = self.variables
        for i in range(other-1):
            variables += self.variables
        return D(*variables)

    @staticmethod
    def deriv(poly, xyz):
        res = 0

        res += Derivative(poly, xyz).doit()

        original = D.diff_symbols.copy()
        original_nc = D.diff_symbols_nc.copy()

        for sym in original:
            deriv_term = Derivative(poly, sym).doit()
            if deriv_term != 0:
                newName = multi_var_deriv_name(sym, xyz)

                dsym = Symbol(newName, commutative=True, real=True)
                
                if dsym not in D.diff_symbols:
                    D.create_diff_symbol(newName)
                    
                res += deriv_term * dsym
                
        for sym in original_nc:
            deriv_term = Derivative(poly, sym).doit()
            if deriv_term != 0:
                newName = multi_var_deriv_name(sym, xyz)

                dsym = Symbol(newName, commutative=False)
                
                if dsym not in D.diff_symbols_nc:
                    D.create_diff_symbol(newName)
                    
                res += deriv_term * dsym
                
        return res
    
    @staticmethod
    def multi_deriv(poly, *variables):
        result = poly
        for xyz in variables:
            result = D.deriv(result, xyz)
        return result
    
    @staticmethod
    def create_diff_symbol(name):
        new_symbol = Symbol(name, commutative=True, real=True)
        if new_symbol not in D.diff_symbols:
            D.diff_symbols.append(new_symbol)
        new_symbol_nc = Symbol(name, commutative=False)
        if new_symbol_nc not in D.diff_symbols_nc:
            D.diff_symbols_nc.append(new_symbol_nc)
        return new_symbol, new_symbol_nc

    @staticmethod
    def create_non_diff_symbol(name):
        new_symbol = Symbol(name, commutative=True, real=True)
        if new_symbol not in D.non_diff_symbols:
            D.non_diff_symbols.append(new_symbol)
        new_symbol_nc = Symbol(name, commutative=False)
        if new_symbol_nc not in D.diff_symbols_nc:
            D.non_diff_symbols_nc.append(new_symbol_nc)
        return new_symbol, new_symbol_nc
    
    @staticmethod
    def create_diff_symbols(*names):
        new_symbols = []
        for name in names:
            new_symbol, new_symbol_nc = D.create_diff_symbol(name)
            new_symbols.append(new_symbol_nc)
            new_symbols.append(new_symbol)

        return new_symbols
    
    @staticmethod
    def create_non_diff_symbols(*names):
        new_symbols = []
        for name in names:
            new_symbol, new_symbol_nc = D.create_non_diff_symbol(name)
            new_symbols.append(new_symbol_nc)
            new_symbols.append(new_symbol)
            
        return new_symbols
    
    @staticmethod
    def comm_to_non_comm(expr):
        for (sym, sym_nc) in zip(D.diff_symbols, D.diff_symbols_nc):
            expr = expr.subs(sym, sym_nc)
        for (sym, sym_nc) in zip(D.non_diff_symbols, D.non_diff_symbols_nc):
            expr = expr.subs(sym, sym_nc)
        return expr
    
    @staticmethod
    def non_comm_to_comm(expr):
        for (sym, sym_nc) in zip(D.diff_symbols, D.diff_symbols_nc):
            expr = expr.subs(sym_nc, sym)
        for (sym, sym_nc) in zip(D.non_diff_symbols, D.non_diff_symbols_nc):
            expr = expr.subs(sym_nc, sym)
        return expr
        
    @staticmethod
    def reset_symbols():
        D.diff_symbols = []
        D.diff_symbols_nc = []
        D.non_diff_symbols = []
        D.non_diff_symbols_nc = []
        

def var_deriv_name(var):
    if ')_{x' in var.name:
        return var.name[:-1] + 'x}'
    else:
        return '(' + var.name + ')_{x}'
        #return var.name + '_x'

def multi_var_deriv_name(var, xyz):
    if ')_{' in var.name:
        i = var.name.rindex(')_{') + 3
        derivs = var.name[i:-1]

        return var.name[:i] + ''.join(sorted(derivs + xyz.name)) + '}'
    else:
        return '(' + var.name + ')_{' + xyz.name + '}'
    
def mydiff(expr, *variables):
    if isinstance(expr, D):
        expr.variables += variables
        return D(*expr.variables)
    if isinstance(expr, Matrix):
        expr_copy = expr.copy()
        for i, elem in enumerate(expr):
            expr_copy[i] = D.multi_deriv(expr, *variables)
        return expr_copy
    if isinstance(expr, conjugate):
        return conjugate(D.multi_deriv(expr.args[0], *variables))
    
    return D.multi_deriv(expr, *variables)


def isFunction(expr):
    return hasattr(expr, 'args') and len(expr.args) > 0 and hasattr(expr, 'func')

spacing = '|   '
def evaluateMul(expr, printing=False, space=spacing, postFunc=None):
    if printing:
        print(space, 'evaluateMul', expr, expr.args)
    if hasattr(expr, 'expand'):
        expr = expr.expand()
    if expr.args:
        if printing:
            print(space, 'hasArgs')
        if isinstance(expr.args[-1], D):
            if printing:
                print(space, 'finalD: zero')
            return Zero()
    initial_args = expr.args
    for i in range(len(expr.args)-1, -1, -1):
        arg = initial_args[i]
        if hasattr(arg, 'expand'):
            arg = arg.expand()
        if printing:
            print(space, 'arg', i, 'is', arg)
        if isinstance(arg, D):
            if printing:
                print(space, 'arg is D')
            left = Mul(*initial_args[:i])
            if printing:
                print(space, 'left', left)
            right = Mul(*expr.args[i+1:])
            if printing:
                print(space, 'right', right)
            right = mydiff(right, *arg.variables)
            if printing:
                print(space, 'new right', right)
            if printing:
                print(space, 'restart')
            return proc(left * right, printing=printing, space=space+spacing, postFunc=postFunc)
        else:
            if printing:
                print(space, 'arg is processed further')
            arg = proc(arg, printing=printing, space=space+spacing, postFunc=postFunc)
            left = Mul(*initial_args[:i])
            if printing:
                print(space, 'left', left)
            right = Mul(*expr.args[i+1:])
            if printing:
                print(space, 'right', right)
            expr = left * arg * right
            if len(expr.args) < len(initial_args):
                return proc(expr, printing=printing, space=space+spacing, postFunc=postFunc)

    if printing:
        print(space, '--Mul-->', expr)
    return postFunc(expr) if postFunc else expr

def proc(expr, printing=False, space=spacing, postFunc=None):
    if hasattr(expr, 'expand'):
        expr = expr.expand()
    if isinstance(expr, Matrix):
        for i, elem in enumerate(expr):
            expr[i] = proc(elem, printing=printing, space=space+spacing, postFunc=postFunc)
    elif isinstance(expr, Mul):
        expr = evaluateMul(expr, printing=printing, space=space+spacing, postFunc=postFunc)
    elif isinstance(expr, D):
        expr = Zero()
    elif isFunction(expr):
        new_args = [proc(a, printing=printing, space=space+spacing, postFunc=postFunc) for a in expr.args]
        expr = expr.func(*new_args)
    return postFunc(expr) if postFunc else expr

def evaluateExpr(expr, printing=False, space=spacing):
    expr = D.comm_to_non_comm(expr)
    expr = proc(expr, printing=printing, space=space)
    expr = D.non_comm_to_comm(expr)
    return(expr)

from collections import deque
from multiset import Multiset

def get_var_name_from_deriv(sym):
    start = sym.name.find('(')
    end = sym.name.find(')')
    if start != -1 and end != -1:
        sym_name = sym.name[start+1:end]
        return sym_name
    else:
        sym_name = sym.name
        return sym_name
        
def get_multiindex_from_deriv(sym):
    start = sym.name.find('(')
    end = sym.name.find(')')
    if start != -1 and end != -1:
        return sym.name[end+3:-1]
    else:
        return ''
    
def get_order_from_deriv(sym):
    start = sym.name.find('(')
    end = sym.name.find(')')
    if start != -1 and end != -1:
        sym_name = sym.name[start+1:end]
        return len(sym.name) - 4 - sym.name.find(')_{')
    else:
        return 0

def deriv(poly):
    return multi_deriv(poly, [x])

def higher_deriv(poly, n):
    return multi_deriv(poly, [x] * n)
   
def multi_deriv(expr, xyz):
    if isinstance(xyz, (list, tuple)):
        result = expr
        for elem in xyz:
            result = multi_deriv(result, elem)
        return result
    
    if isinstance(expr, Matrix):
        expr_copy = expr.copy()
        for i, elem in enumerate(expr):
            expr_copy[i] = multi_deriv(elem, xyz)
        return expr_copy
    
    res = 0
    res += Derivative(expr, xyz).doit()

    fixed_symbols = (set(D.diff_symbols) | set(D.diff_symbols_nc)) & set(expr.free_symbols)
    symbols_to_iterate = list(fixed_symbols)  # Creates a separate list copy

    for sym in symbols_to_iterate:
        deriv_term = Derivative(expr, sym).doit()
        if deriv_term != 0:
            newName = multi_var_deriv_name(sym, xyz)
            if sym.is_commutative:
                dsym = next((s for s in D.diff_symbols if s.name == newName), D.create_diff_symbols(newName)[1])
            else:
                dsym = next((s for s in D.diff_symbols_nc if s.name == newName), D.create_diff_symbols(newName)[0])
            res += deriv_term * dsym
    
    return res

def single_subs(expr, var, sub, scale=1):
    if not hasattr(expr, 'subs'):
        return expr
        
    if isinstance(scale, dict):
        for xyz, scaling in scale.items():
            expr = expr.subs(xyz, scaling * xyz)
            
    if var not in set(D.diff_symbols) | set(D.diff_symbols_nc):
        return expr.subs(var, sub)
    
    var_name = get_var_name_from_deriv(var)
    var_multiindex = Multiset(get_multiindex_from_deriv(var))
    
    fixed_symbols = (set(D.diff_symbols) | set(D.diff_symbols_nc)) & set(expr.free_symbols)
    symbols_to_iterate = list(fixed_symbols)  # Creates a separate list copy

    for sym in symbols_to_iterate:
        if get_var_name_from_deriv(sym) == var_name:
            sym_multiindex = Multiset(get_multiindex_from_deriv(sym))
            if var_multiindex.issubset(sym_multiindex):
                target_multiindex = sym_multiindex.difference(var_multiindex)
                target_operator = [Symbol(char, real=True) for char in target_multiindex]
                if isinstance(scale, dict):
                    factor = 1
                    for direction in target_operator:
                        factor *= scale.get(direction, 1)
                else:
                    factor = scale**len(target_operator)
                expr = expr.subs(sym, factor * multi_deriv(sub, target_operator))

    return expr

def subs(expr, data, scale=1):
    for (var, sub) in data:
        expr = single_subs(expr, var, sub, scale=scale)
    return expr

def variation(expr, sym):
    if not hasattr(expr, 'free_symbols'):
        return 0
    
    res = 0
    order = 0

    syms = []
    orders = []

    start = sym.name.find('(')
    end = sym.name.find(')')
    if start != -1 and end != -1:
        sym_name = sym.name[start+1:end]
        syms.append(sym)
        orders.append(len(sym.name) - 4 - sym.name.find(')_{'))
    else:
        sym_name = sym.name
        syms.append(sym)
        orders.append(0)

    for s in expr.free_symbols:
        start = s.name.find('(')
        end = s.name.find(')')
        if start != -1 and end != -1 and sym_name == s.name[start+1:end]:
            if s.name not in [sym.name for sym in syms]:
                syms.append(s)
                orders.append(len(s.name) - 4 - s.name.find(')_{'))    
    
    for (sym, order) in zip(syms, orders):
        res += (-1)**order * higher_deriv(Derivative(expr, sym).doit(), order)

    return simplify(res)


###############################################
# Create symbols and define PDEs & Lax Pairs  #
###############################################
# D.reset_symbols()
# x, xx, y, yy, z, zz, tt, t = D.create_non_diff_symbols('x', 'y', 'z', 't')
# q, qq, q_conj, qq_conj = D.create_diff_symbols('q', '\\tilde{q}')
# u, uu, v, vv, a, aa, phi, pphi, f, ff, g, gg = D.create_diff_symbols('u', 'v', 'a', '\phi', '\\tilde{\\hat{f}}', 'g')

# # Example 1: The KdV Equation
# KdV_LHS = multi_deriv(uu, t)
# KdV_RHS = 6 * uu * deriv(uu) - higher_deriv(uu, 3)

# L_KdV = - D(x, x) + u
# P_KdV = - 4 * D(x, x, x) + 3 * evaluateExpr(D(x) * u) + 6 * u * D(x)

# tested_KdV_Lax_equation = evaluateExpr((evaluateExpr(D(tt) * L_KdV) + L_KdV * P_KdV - P_KdV * L_KdV) * f)
# tested_KdV_equation = (KdV_LHS - KdV_RHS) * ff

# display(Equ(Symbol('\\text{Error}_{\\text{KdV vs Lax pair}}'), simplify(tested_KdV_Lax_equation - tested_KdV_equation)))

# # Example 2: The Focusing NLS Equation
# NLS_q_LHS = I * multi_deriv(qq, t)
# NLS_q_RHS = - higher_deriv(qq, 2) + 2 * qq**2 * qq_conj
# NLS_q_conj_LHS = I * multi_deriv(qq_conj, t)
# NLS_q_conj_RHS = higher_deriv(qq_conj, 2) - 2 * qq_conj**2 * qq

# L_NLS = I * Matrix([[D(x), - q], [q_conj, - D(x)]])
# P_NLS = I * Matrix([[2 * D(x, x) - q * q_conj, - q * D(x) - D(x) * q],
#                     [q_conj * D(x) + D(x) * q_conj, - 2 * D(x, x) + q * q_conj]])

# tested_NLS_Lax_equation = evaluateExpr((proc(D(tt) * L_NLS) + L_NLS * P_NLS - P_NLS * L_NLS) * f)
# tested_NLS_equation = Matrix([[0, NLS_q_RHS - NLS_q_LHS], [- NLS_q_conj_RHS + NLS_q_conj_LHS, 0]]) * ff

# display(Equ(Symbol('\\text{Error}_{\\text{NLS vs Lax pair}}'), simplify(tested_NLS_Lax_equation - tested_NLS_equation)))


D.reset_symbols()
a, _, b, _, f, _, g, _, h, _, p, _, q, _, r, _, s, _, u, _, v, _, w, _ = D.create_diff_symbols('a', 'b', 'f', 'g', 'h', 'p', 'q', 'r', 's', 'u', 'v', 'w')
t, _, x, _, y, _, z, _ = D.create_non_diff_symbols('t', 'x', 'y', 'z')
c, d, e, i, j, k, l, m, n, o = symbols("c, d, e, i, j, k, l, m, n, o", real=True)
test, ttest = D.create_diff_symbols('\\tilde{f}')

def parse_expression(code_str, extra_env=None):
    base_env = {
        "D": D,
        "sin": sin,
        "cos": cos,
        "exp": exp,
        "cosh": cosh,
        "sinh": sinh,
        "I": I,
        "simplify": simplify,
        "Matrix": Matrix,
        "Mul": Mul,
        "Add": Add,
        "Eq": Eq,
        "Symbol": Symbol,
        "Derivative": Derivative,
        "a": a, "b": b, "f": f, "g": g, "h": h, "p": p, "q": q, "r": r, "s": s, "u": u, "v": v, "w": w,
        "t": t, "x": x, "y": y, "z": z,
        "c": c, "d": d, "e": e, "i": i, "j": j, "k": k, "l": l, "m": m, "n": n, "o": o
    }

    if extra_env:
        base_env.update(extra_env)

    return eval(code_str, {}, base_env)

def get_Lax_equation(L, P):
    return simplify(evaluateExpr((proc(D(t) * L) + L * P - P * L) * test) / ttest)

def test_Lax_equation(Lax_equation):
    return ttest not in Lax_equation.free_symbols

def check_Lax_pair(L_code, P_code):
    L_expr = parse_expression(L_code)
    P_expr = parse_expression(P_code)
    try:
        Lax_equation = get_Lax_equation(L_expr, P_expr)
        is_Lax_pair = test_Lax_equation(Lax_equation)
        return is_Lax_pair, Lax_equation, latex(Lax_equation), latex(L_expr), latex(P_expr)
    except Exception as error:
        print("Error while checking Lax pair:", error)
        traceback.print_exc()
        return error, False
    