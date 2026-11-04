#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
선형대수 5장(직교) 연습문제 생성기.

    python generator.py --seed 1 --count 48            # index.html 한 장
    python generator.py --seed 1 --count 48 --sets 3   # index.html, set-2.html, set-3.html

모든 문항은 verify.py 의 SymPy 검산을 통과한 것만 출력된다.
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass, field
from fractions import Fraction as Fr
from functools import lru_cache, reduce
from itertools import product
from pathlib import Path

import sympy as sp

SUP = {2: "²", 3: "³", 4: "⁴"}
HERE = Path(__file__).resolve().parent
ALLOWED_RADICANDS = (1, 2, 3, 5, 6)
LAM = sp.Symbol("lambda")
NSYM = sp.Symbol("n", integer=True, positive=True)
MATHJAX_URL = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-mml-chtml.min.js"

# (키, 문자, 제목)
KINDS = [
    ("complement", "A", "직교여공간"),
    ("projection", "B", "정사영과 직교분해"),
    ("gram_schmidt", "C", "그램-슈미트 과정"),
    ("qr", "D", "QR 분해"),
    ("diag", "E", "대칭행렬의 직교대각화"),
    ("spectral", "F", "스펙트럼 분해"),
    ("inverse", "G", "역문제: 고유정보로 행렬 구하기"),
    ("truefalse", "H", "참/거짓 개념 확인"),
]


@dataclass
class Problem:
    kind: str
    tag: str
    statement: str
    steps: list
    answer: str
    data: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 정수/벡터 도우미
# ---------------------------------------------------------------------------
def dot(u, v):
    return sum(a * b for a, b in zip(u, v))


def nnz(v):
    return sum(1 for x in v if x != 0)


def sqfree(N):
    """N = d^2 * r (r 은 제곱인수 없음) 인 (d, r)."""
    for d in range(math.isqrt(N), 0, -1):
        if N % (d * d) == 0:
            return d, N // (d * d)


def nice_norm2(N, maxN=36):
    return 0 < N <= maxN and sqfree(N)[1] in ALLOWED_RADICANDS


@lru_cache(maxsize=None)
def prim_vectors(n, bound):
    """성분이 [-bound, bound] 인 원시 정수벡터(첫 0 아닌 성분 > 0)."""
    out = []
    for v in product(range(-bound, bound + 1), repeat=n):
        if not any(v) or math.gcd(*v) != 1:
            continue
        if next(x for x in v if x) < 0:
            continue
        out.append(v)
    return out


def orthogonal_set(rng, n, k, bound=2, min_nnz=2, maxN=36):
    """노름의 제곱이 '깔끔한' 서로 직교하는 정수벡터 k개."""
    pool = [v for v in prim_vectors(n, bound) if nice_norm2(dot(v, v), maxN) and nnz(v) >= min_nnz]
    for _ in range(500):
        chosen = []
        for _i in range(k):
            c = [v for v in pool if all(dot(v, w) == 0 for w in chosen)]
            if not c:
                break
            chosen.append(rng.choice(c))
        if len(chosen) == k:
            return chosen
    raise RuntimeError("직교 집합을 찾지 못함")


def col(v):
    return sp.Matrix([sp.nsimplify(x) for x in v])


def to_int_primitive(v):
    """유리수 벡터 v -> (원시 정수 튜플 w, 양수 f) 이고 w = f*v."""
    vals = [sp.Rational(x) for x in v]
    L = reduce(math.lcm, [int(x.q) for x in vals], 1)
    ints = [int(x * L) for x in vals]
    g = math.gcd(*ints)
    return tuple(i // g for i in ints), sp.Rational(L, g)


def fr_to_sp(x):
    return sp.Rational(x.numerator, x.denominator)


# ---------------------------------------------------------------------------
# LaTeX 도우미
# ---------------------------------------------------------------------------
def lt(x):
    return sp.latex(sp.nsimplify(x) if not isinstance(x, sp.Basic) else x)


def tup(v):
    return r"\left(" + r",\, ".join(lt(x) for x in v) + r"\right)"


def bmat(Mx):
    rows = [" & ".join(lt(e) for e in Mx.row(i)) for i in range(Mx.rows)]
    return r"\begin{bmatrix}" + r" \\ ".join(rows) + r"\end{bmatrix}"


def frac_mat(N, Mx):
    return ("" if N == 1 else rf"\frac{{1}}{{{N}}}") + bmat(Mx)


def unit_coef(N):
    d, r = sqfree(N)
    if r == 1:
        return "" if d == 1 else rf"\frac{{1}}{{{d}}}"
    return rf"\frac{{1}}{{{'' if d == 1 else d}\sqrt{{{r}}}}}"


def unit_tex(w):
    return unit_coef(dot(w, w)) + tup(w)


def lincomb(base_tex, terms):
    out = base_tex
    for c, tex in terms:
        if c == 0:
            continue
        sign = "-" if c > 0 else "+"
        a = abs(c)
        out += f" {sign} " + ("" if a == 1 else lt(a)) + tex
    return out


def signed_comb(terms):
    """[(c, tex)] -> c1 tex1 + c2 tex2 ..."""
    out = ""
    for c, tex in terms:
        if c == 0:
            continue
        a = abs(c)
        coef = "" if a == 1 else lt(a)
        if coef and (tex.startswith("\\frac") or tex[:1].isdigit()):
            coef += r"\cdot "
        if not out:
            out = ("-" if c < 0 else "") + coef + tex
        else:
            out += (" - " if c < 0 else " + ") + coef + tex
    return out or "0"


def eq_tex(coeffs):
    terms = [(sp.Integer(c), f"x_{{{i + 1}}}") for i, c in enumerate(coeffs)]
    return signed_comb(terms) + " = 0"


def bv(name, i=None):
    return rf"\mathbf{{{name}}}" + ("" if i is None else f"_{{{i}}}")


def m(tex):
    return rf"\({tex}\)"


def disp(tex):
    return rf"\[{tex}\]"


def mlist(items, sep=", "):
    """수식을 하나씩 따로 감싸 휴대폰에서 줄바꿈되게 한다."""
    return sep.join(m(x) for x in items)


def lam_paren(lam):
    return lt(lam) if lam >= 0 else f"({lt(lam)})"


# ---------------------------------------------------------------------------
# 공통 풀이 조각
# ---------------------------------------------------------------------------
def null_steps(A, var="x"):
    """Ax=0 풀이 단계와 정수 기저."""
    R, piv = A.rref()
    n = A.cols
    free = [j for j in range(n) if j not in piv]
    params = ["s", "t", "u"][: len(free)]
    raw = A.nullspace()
    ints = [to_int_primitive(b)[0] for b in raw]
    steps = [disp(bmat(A) + r"\;\sim\;" + bmat(R))]
    fv = ",\\; ".join(f"{var}_{{{j + 1}}} = {p}" for j, p in zip(free, params))
    gen = " + ".join(f"{p}{tup(b)}" for p, b in zip(params, raw))
    steps.append(f"자유변수를 {m(fv)} 로 두면 {m(rf'{bv(var)} = ' + gen)} 이다.")
    if any(tuple(b) != w for b, w in zip(raw, ints)):
        steps.append("분수를 없애기 위해 각 벡터를 상수배하면 기저 " + m(r"\left\{" + ",\\ ".join(tup(w) for w in ints) + r"\right\}") + " 를 얻는다.")
    return ints, steps


def gs_steps(X, xs="x", vs="v"):
    """그램-슈미트 (중간 결과는 정수배로 정리). 반환: (정수 직교벡터들, 단계, 원래 v들)."""
    V, steps, raws = [], [], []
    for k, x in enumerate(X):
        i = k + 1
        xn, vn = bv(xs, i), bv(vs, i)
        if k == 0:
            V.append(tuple(x))
            raws.append(col(x))
            steps.append(f"{m(f'{vn} = {xn} = {tup(x)}')}")
            continue
        v = col(x)
        terms, dots = [], []
        formula = xn
        for j, w in enumerate(V):
            wn = bv(vs, j + 1)
            a, b = dot(x, w), dot(w, w)
            c = sp.Rational(a, b)
            dots.append(f"{xn}\\cdot {wn} = {a},\\ {wn}\\cdot {wn} = {b}")
            terms.append((c, tup(w)))
            formula += rf" - \frac{{{xn}\cdot {wn}}}{{{wn}\cdot {wn}}}{wn}"
            v = v - c * col(w)
        raws.append(v)
        steps.append(mlist(dots) + " 이므로")
        steps.append(disp(rf"\begin{{aligned}}{vn} &= {formula} \\ &= {lincomb(tup(x), terms)} = {tup(v)}\end{{aligned}}"))
        ints, f = to_int_primitive(v)
        if f != 1:
            steps.append(f"직교성은 상수배해도 유지되므로 {m(f'{lt(f)}{vn} = {tup(ints)}')} 를 새로 {m(vn)} 으로 쓴다.")
        V.append(ints)
    return V, steps, raws


def charpoly(A):
    p = sp.expand((A - LAM * sp.eye(A.rows)).det())
    return p, sp.factor(p)


def eig_list_tex(roots):
    parts = []
    for lam in sorted(roots, reverse=True):
        parts.append(lt(lam) + (r"\ (\text{중근})" if roots[lam] == 2 else (r"\ (\text{삼중근})" if roots[lam] == 3 else "")))
    return r",\ ".join(parts)


def diag_solution(A, name="A"):
    """A 를 SymPy 로 직접 분석해 직교대각화 풀이를 만든다."""
    n = A.rows
    steps = []
    pe, pf = charpoly(A)
    roots = sp.roots(sp.Poly(pe, LAM))
    steps.append(
        "<b>① 고윳값.</b> 특성다항식은"
        + disp(rf"\begin{{aligned}}\det({name}-\lambda I) &= {lt(pe)} \\ &= {lt(pf)}\end{{aligned}}")
        + f"이므로 고윳값은 {m(r'\lambda = ' + eig_list_tex(roots))} 이다."
    )
    eig = []
    idx = 1
    for lam in sorted(roots, reverse=True):
        mult = roots[lam]
        Mx = A - lam * sp.eye(n)
        ints, nsteps = null_steps(Mx)
        head = f"<b>② {m(r'\lambda = ' + lt(lam))} 의 고유공간.</b> {m(f'{name} - {lam_paren(lam)}I')} 를 행간소화하면"
        steps.append(head + nsteps[0])
        steps.extend(nsteps[1:])
        if mult >= 2 and any(dot(ints[i], ints[j]) != 0 for i in range(len(ints)) for j in range(i)):
            steps.append(
                f"이 고유공간의 기저 {m(', '.join(tup(w) for w in ints))} 는 서로 직교하지 않으므로 "
                "<b>이 고유공간 안에서만</b> 그램-슈미트 과정을 적용한다. "
                f"{m(', '.join(f'{bv('b', i + 1)} = {tup(w)}' for i, w in enumerate(ints)))} 라 하면"
            )
            vecs, gsteps, _ = gs_steps(ints, xs="b", vs="p")
            steps.extend(gsteps)
            steps.append(
                f"서로 다른 고윳값의 고유벡터는 자동으로 직교하므로, 이제 직교 고유벡터 {m(', '.join(tup(w) for w in vecs))} 를 얻었다."
            )
        else:
            vecs = ints
        eig.append((lam, vecs))
    cols, dvals, qtex = [], [], []
    for lam, vecs in eig:
        for w in vecs:
            cols.append(col(w) / sp.sqrt(dot(w, w)))
            dvals.append(lam)
            qtex.append(f"{bv('q', idx)} = {unit_tex(w)}")
            idx += 1
    P = sp.Matrix.hstack(*cols)
    D = sp.diag(*dvals)
    steps.append("<b>③ 정규화.</b> 각 고유벡터를 단위벡터로 만들면 " + mlist(qtex) + ".")
    steps.append(
        "<b>④ 결론.</b>"
        + disp(rf"P = \begin{{bmatrix}} {' & '.join(bv('q', i + 1) for i in range(n))} \end{{bmatrix}} = {bmat(P)}")
        + disp(f"D = {bmat(D)}")
        + f"이면 {m('P^TP = I')} 이고 {m(f'P^T{name}P = D')}, 즉 {m(f'{name} = PDP^T')} 이다."
    )
    return dict(P=P, D=D, eig=eig, steps=steps, roots=roots, charpoly=pf)


def sym_from_eig(eig, n):
    A = [[Fr(0)] * n for _ in range(n)]
    for lam, vs in eig:
        for w in vs:
            N = dot(w, w)
            for i in range(n):
                for j in range(n):
                    A[i][j] += Fr(lam * w[i] * w[j], N)
    return A


def is_int_small(A, max_entry):
    return all(x.denominator == 1 and abs(x) <= max_entry for row in A for x in row)


def is_diagonal(A):
    return all(A[i][j] == 0 for i in range(len(A)) for j in range(len(A)) if i != j)


def nice_symmetric(rng, kind, lam_range=range(-3, 9), max_entry=9, exclude=()):
    """kind: '2' (2x2), '3d' (3x3 서로 다른 고윳값), '3r' (3x3 중근). (A, eig) 반환."""
    lams_ok = [x for x in lam_range if x not in exclude]
    for _ in range(300):
        if kind == "2":
            u = orthogonal_set(rng, 2, 1, bound=4)[0]
            w = (-u[1], u[0]) if u[1] > 0 else (u[1], -u[0])
            vecs = [u, w]
            if rng.random() < 0.5:
                vecs = [w if w[0] > 0 else tuple(-x for x in w), u]
            cands = [(a, b) for a in lams_ok for b in lams_ok if a != b]
            rng.shuffle(cands)
            for a, b in cands:
                eig = [(a, [vecs[0]]), (b, [vecs[1]])]
                A = sym_from_eig(eig, 2)
                if is_int_small(A, max_entry) and not is_diagonal(A):
                    return sp.Matrix(2, 2, [fr_to_sp(x) for r in A for x in r]), eig
        elif kind == "3d":
            W = orthogonal_set(rng, 3, 3, bound=2)
            cands = [t for t in product(lams_ok, repeat=3) if len(set(t)) == 3]
            rng.shuffle(cands)
            for t in cands:
                eig = [(t[i], [W[i]]) for i in range(3)]
                A = sym_from_eig(eig, 3)
                if is_int_small(A, max_entry) and not is_diagonal(A):
                    return sp.Matrix(3, 3, [fr_to_sp(x) for r in A for x in r]), eig
        else:  # '3r'
            pool = [v for v in prim_vectors(3, 2) if nnz(v) == 3 and nice_norm2(dot(v, v))]
            u = rng.choice(pool)
            N = dot(u, u)
            cands = [(a, b) for a in lams_ok for b in lams_ok if a != b]
            rng.shuffle(cands)
            for lr, ls in cands:
                A = [[Fr(lr if i == j else 0) + Fr((ls - lr) * u[i] * u[j], N) for j in range(3)] for i in range(3)]
                if is_int_small(A, max_entry):
                    return sp.Matrix(3, 3, [fr_to_sp(x) for r in A for x in r]), [(lr, None), (ls, [u])]
    raise RuntimeError("깔끔한 대칭행렬 생성 실패")


# ---------------------------------------------------------------------------
# A. 직교여공간
# ---------------------------------------------------------------------------
COMPLEMENT_VARIANTS = [(3, 2, "span"), (4, 2, "span"), (3, 1, "span"), (4, 2, "eq"), (4, 3, "span"), (4, 1, "eq")]


def gen_complement(rng, i):
    n, k, mode = COMPLEMENT_VARIANTS[i % len(COMPLEMENT_VARIANTS)]
    while True:
        rows = [tuple(rng.randint(-2, 2) for _ in range(n)) for _ in range(k)]
        if any(nnz(r) < 2 for r in rows):
            continue
        A = sp.Matrix(rows)
        if A.rank() < k:
            continue
        R, _ = A.rref()
        if any(e.q > 2 or abs(e.p) > 4 for e in R):
            continue
        ints = [to_int_primitive(b)[0] for b in A.nullspace()]
        if max(abs(x) for v in ints for x in v) > 6:
            continue
        break
    Rn = rf"\mathbb{{R}}^{n}"
    if mode == "span":
        ws = ", ".join(f"{bv('w', j + 1)} = {tup(r)}" for j, r in enumerate(rows))
        span = ", ".join(bv("w", j + 1) for j in range(k))
        st = f"{m(Rn)} 의 부분공간 {m(rf'W = \operatorname{{span}}\{{{span}\}}')} 에 대하여 {m(r'W^\perp')} 의 기저와 차원을 구하여라. 단, {m(ws)}."
        ints, ns = null_steps(A)
        steps = [
            f"{m(r'\mathbf{x} \in W^\perp')} 이려면 {m(r'\mathbf{x}')} 가 생성벡터 모두와 직교해야 한다: "
            + m(r",\ ".join(f"{bv('w', j + 1)}\\cdot\\mathbf{{x}} = 0" for j in range(k)))
            + f". 즉 {m(bv('w', 1) + (', …' if k > 1 else ''))} 를 행으로 하는 행렬 {m('A')} 에 대해 {m(r'W^\perp = \operatorname{Null}(A)')} 이다.",
            "행간소화:" + ns[0],
            *ns[1:],
            f"확인: {m(r'\dim W + \dim W^\perp = ' + f'{k} + {n - k} = {n}')}.",
        ]
        ans = f"{m(r'W^\perp')} 의 기저: {m(r'\left\{' + r',\ '.join(tup(w) for w in ints) + r'\right\}')}, {m(r'\dim W^\perp = ' + str(n - k))}"
        ans += " <span class='note'>(기저는 유일하지 않다. 같은 공간을 생성하면 정답)</span>"
        data = dict(W_basis=A.T, perp=[col(w) for w in ints])
        tag = f"ℝ{SUP[n]}, dim W = {k}"
    else:
        eqs = r",\ ".join(eq_tex(r) for r in rows)
        st = f"{m(Rn)} 의 부분공간 {m(rf'W = \{{\mathbf{{x}}\in{Rn} : {eqs}\}}')} 에 대하여 {m('W')} 와 {m(r'W^\perp')} 의 기저를 각각 구하여라."
        ints, ns = null_steps(A)
        rowtex = r",\ ".join(tup(r) for r in rows)
        steps = [
            f"계수 행렬을 {m('A')} 라 하면 {m(r'W = \operatorname{Null}(A)')} 이다. 따라서 {m(r'W^\perp = (\operatorname{Null} A)^\perp = \operatorname{Row}(A)')}.",
            f"{m('A')} 의 행 {m(rowtex)} 는 일차독립이므로(계수 {k}) 그대로 {m(r'W^\perp')} 의 기저가 된다.",
            f"{m('W')} 의 기저는 {m(r'A\mathbf{x} = \mathbf{0}')} 을 풀어 구한다:" + ns[0],
            *ns[1:],
            f"확인: {m(r'\dim W + \dim W^\perp = ' + f'{n - k} + {k} = {n}')}.",
        ]
        ans = (
            f"{m('W')} 의 기저: {m(r'\left\{' + r',\ '.join(tup(w) for w in ints) + r'\right\}')}, "
            f"{m(r'W^\perp')} 의 기저: {m(r'\left\{' + rowtex + r'\right\}')}"
        )
        data = dict(W_basis=sp.Matrix.hstack(*[col(w) for w in ints]), perp=[col(r) for r in rows])
        tag = f"ℝ{SUP[n]}, 방정식으로 주어진 W"
    return Problem("complement", tag, st, steps, ans, data)


# ---------------------------------------------------------------------------
# B. 정사영과 직교분해
# ---------------------------------------------------------------------------
PROJECTION_VARIANTS = ["plane", "orth3", "nonorth3", "orth4", "nonorth4", "plane"]


def _rand_combo(rng, basis, choices):
    cs = [rng.choice(choices) for _ in basis]
    n = len(basis[0])
    return cs, tuple(sum(Fr(c) * b[i] for c, b in zip(cs, basis)) for i in range(n))


def gen_projection(rng, i):
    var = PROJECTION_VARIANTS[i % len(PROJECTION_VARIANTS)]
    if var == "plane":
        while True:
            nvec = rng.choice([v for v in prim_vectors(3, 2) if nnz(v) >= 2 and nice_norm2(dot(v, v))])
            nvec = tuple(x * rng.choice([1, -1]) if j else x for j, x in enumerate(nvec))
            if not nice_norm2(dot(nvec, nvec)):
                continue
            pb = [to_int_primitive(b)[0] for b in sp.Matrix([nvec]).nullspace()]
            _, w = _rand_combo(rng, pb, [-2, -1, 1, 2])
            t = rng.choice([-2, -1, 1, 2])
            v = tuple(int(w[j] + t * nvec[j]) for j in range(3))
            if max(map(abs, v)) <= 9 and nnz(v) >= 2 and nnz(w) >= 1:
                break
        N = dot(nvec, nvec)
        vn = dot(v, nvec)
        perp = col(nvec) * sp.Rational(vn, N)
        proj = col(v) - perp
        plane = signed_comb([(sp.Integer(c), s) for c, s in zip(nvec, "xyz")]) + " = 0"
        st = (
            f"평면 {m(f'W : {plane}')} 와 벡터 {m(r'\mathbf{v} = ' + tup(v))} 에 대하여 "
            f"{m(r'\mathbf{v} = \mathbf{w} + \mathbf{z}')} ({m(r'\mathbf{w}\in W,\ \mathbf{z}\in W^\perp')}) 로 직교분해하고, "
            f"점 {m(tup(v))} 와 평면 {m('W')} 사이의 거리를 구하여라."
        )
        steps = [
            f"평면의 법선벡터는 {m(r'\mathbf{n} = ' + tup(nvec))} 이고 {m(r'W^\perp = \operatorname{span}\{\mathbf{n}\}')} 이다.",
            f"{m(r'\mathbf{v}\cdot\mathbf{n} = ' + str(vn))}, {m(r'\mathbf{n}\cdot\mathbf{n} = ' + str(N))} 이므로"
            + disp(rf"\mathbf{{z}} = \operatorname{{proj}}_{{W^\perp}}\mathbf{{v}} = \frac{{\mathbf{{v}}\cdot\mathbf{{n}}}}{{\mathbf{{n}}\cdot\mathbf{{n}}}}\mathbf{{n}} = {lt(sp.Rational(vn, N))}{tup(nvec)} = {tup(perp)}"),
            disp(rf"\mathbf{{w}} = \operatorname{{proj}}_W\mathbf{{v}} = \mathbf{{v}} - \mathbf{{z}} = {tup(v)} - {tup(perp)} = {tup(proj)}"),
            f"검산: {m(r'\mathbf{w}\cdot\mathbf{n} = ' + lt(proj.dot(col(nvec))))} 이므로 {m(r'\mathbf{w}\in W')}.",
            f"거리 {m(r'= \|\mathbf{z}\| = ' + lt(sp.sqrt(perp.dot(perp))))}.",
        ]
        ans = f"{m(r'\mathbf{w} = ' + tup(proj))}, {m(r'\mathbf{z} = ' + tup(perp))}, 거리 {m(lt(sp.sqrt(perp.dot(perp))))}"
        data = dict(B=sp.Matrix.hstack(*[col(b) for b in pb]), v=col(v), proj=proj, perp=perp)
        return Problem("projection", "ℝ³ 평면", st, steps, ans, data)

    n = 3 if var.endswith("3") else 4
    orth = var.startswith("orth")
    while True:
        if orth:
            U = orthogonal_set(rng, n, 2, bound=2)
            U = [tuple(x * s for x in u) for u, s in zip(U, [rng.choice([1, -1]) for _ in U])]
        else:
            U = [tuple(rng.randint(-2, 2) for _ in range(n)) for _ in range(2)]
            if any(nnz(u) < 2 for u in U) or dot(U[0], U[1]) == 0:
                continue
        B = sp.Matrix.hstack(*[col(u) for u in U])
        if B.rank() < 2:
            continue
        zb = [to_int_primitive(b)[0] for b in B.T.nullspace()]
        _, z = _rand_combo(rng, zb, [-1, 1] if len(zb) > 1 else [-2, -1, 1, 2])
        z = to_int_primitive(z)[0] if len(zb) > 1 else tuple(int(x) for x in z)
        if not nice_norm2(dot(z, z), 50) or not any(z):
            continue
        cs, w = _rand_combo(rng, U, [-2, -1, 1, 2] + ([Fr(1, 2), Fr(-1, 2), Fr(3, 2)] if orth else []))
        v = tuple(w[j] + z[j] for j in range(n))
        if all(x.denominator == 1 for x in v) and max(abs(x) for x in v) <= 9 and nnz(v) >= 2:
            v = tuple(int(x) for x in v)
            break
    proj = sp.Matrix([fr_to_sp(x) for x in w])
    perp = col(v) - proj
    us = ", ".join(f"{bv('u', j + 1)} = {tup(u)}" for j, u in enumerate(U))
    st = (
        f"{m(rf'\mathbb{{R}}^{n}')} 의 부분공간 {m(r'W = \operatorname{span}\{\mathbf{u}_1, \mathbf{u}_2\}')} ({m(us)}) 와 "
        f"{m(r'\mathbf{v} = ' + tup(v))} 에 대하여 {m(r'\operatorname{proj}_W\mathbf{v}')} 와 "
        f"{m(r'\mathbf{v} - \operatorname{proj}_W\mathbf{v}')} 를 구하고, {m(r'\mathbf{v}')} 에서 {m('W')} 까지의 거리를 구하여라."
    )
    if orth:
        a = [dot(v, u) for u in U]
        b = [dot(u, u) for u in U]
        steps = [
            f"{m(r'\mathbf{u}_1\cdot\mathbf{u}_2 = 0')} 이므로 {m(r'\{\mathbf{u}_1,\mathbf{u}_2\}')} 는 <b>직교기저</b>이고 정사영 공식을 바로 쓸 수 있다.",
            m(rf"\mathbf{{v}}\cdot\mathbf{{u}}_1 = {a[0]},\ \mathbf{{u}}_1\cdot\mathbf{{u}}_1 = {b[0]},\ \mathbf{{v}}\cdot\mathbf{{u}}_2 = {a[1]},\ \mathbf{{u}}_2\cdot\mathbf{{u}}_2 = {b[1]}"),
            disp(
                rf"\begin{{aligned}}\operatorname{{proj}}_W\mathbf{{v}} &= \frac{{\mathbf{{v}}\cdot\mathbf{{u}}_1}}{{\mathbf{{u}}_1\cdot\mathbf{{u}}_1}}\mathbf{{u}}_1 + \frac{{\mathbf{{v}}\cdot\mathbf{{u}}_2}}{{\mathbf{{u}}_2\cdot\mathbf{{u}}_2}}\mathbf{{u}}_2"
                rf"\\ &= {signed_comb([(sp.Rational(a[0], b[0]), tup(U[0])), (sp.Rational(a[1], b[1]), tup(U[1]))])} = {tup(proj)}\end{{aligned}}"
            ),
        ]
    else:
        G = B.T * B
        rhs = B.T * col(v)
        xhat = G.solve(rhs)
        steps = [
            f"{m(r'\mathbf{u}_1\cdot\mathbf{u}_2 = ' + str(dot(U[0], U[1])) + r' \neq 0')} 이므로 직교기저가 아니다. "
            f"{m(r'B = [\,\mathbf{u}_1\ \mathbf{u}_2\,]')} 로 두고 정규방정식 {m(r'B^TB\hat{\mathbf{x}} = B^T\mathbf{v}')} 를 푼다. "
            "<span class='note'>(그램-슈미트로 직교기저를 먼저 만들어도 된다.)</span>",
            disp(rf"B^TB = {bmat(G)},\quad B^T\mathbf{{v}} = {bmat(rhs)}"),
            disp(rf"\hat{{\mathbf{{x}}}} = {bmat(xhat)}"),
            disp(rf"\begin{{aligned}}\operatorname{{proj}}_W\mathbf{{v}} = B\hat{{\mathbf{{x}}}} &= {signed_comb([(xhat[0], tup(U[0])), (xhat[1], tup(U[1]))])} \\ &= {tup(proj)}\end{{aligned}}"),
        ]
    dist = sp.sqrt(perp.dot(perp))
    steps += [
        disp(rf"\mathbf{{v}} - \operatorname{{proj}}_W\mathbf{{v}} = {tup(v)} - {tup(proj)} = {tup(perp)}"),
        f"검산: {m(r'(\mathbf{v}-\operatorname{proj}_W\mathbf{v})\cdot\mathbf{u}_1 = ' + lt(perp.dot(col(U[0]))))}, "
        f"{m(r'(\mathbf{v}-\operatorname{proj}_W\mathbf{v})\cdot\mathbf{u}_2 = ' + lt(perp.dot(col(U[1]))))} 로 {m('W')} 에 수직이다.",
        f"거리 {m(r'= \|\mathbf{v}-\operatorname{proj}_W\mathbf{v}\| = ' + lt(dist))}.",
    ]
    ans = f"{m(r'\operatorname{proj}_W\mathbf{v} = ' + tup(proj))}, {m(r'\mathbf{v}-\operatorname{proj}_W\mathbf{v} = ' + tup(perp))}, 거리 {m(lt(dist))}"
    data = dict(B=B, v=col(v), proj=proj, perp=perp)
    tag = f"ℝ{SUP[n]}, " + ("직교기저" if orth else "직교하지 않는 기저")
    return Problem("projection", tag, st, steps, ans, data)


# ---------------------------------------------------------------------------
# C, D. 그램-슈미트 / QR
# ---------------------------------------------------------------------------
GS_VARIANTS = [(3, 2), (3, 3), (4, 2), (4, 3), (3, 3), (4, 3)]
QR_VARIANTS = [(3, 2), (4, 3), (3, 2), (4, 3), (4, 3), (3, 2)]
_DIAG = [Fr(1), Fr(1), Fr(1, 2), Fr(1, 3), Fr(2)]
_OFF = [Fr(0), Fr(1), Fr(-1), Fr(2), Fr(-2), Fr(1, 2), Fr(-1, 2), Fr(3, 2), Fr(1, 3), Fr(-2, 3)]


def gs_input(rng, n, k):
    """직교 정수기저 W 에 상삼각 변환을 걸어 그램-슈미트 결과가 깔끔한 입력 X 를 만든다."""
    for _ in range(20000):
        W = orthogonal_set(rng, n, k, bound=2)
        X = [W[0]]
        ok = True
        for j in range(1, k):
            d = rng.choice(_DIAG)
            offs = [rng.choice(_OFF) for _ in range(j)]
            if all(o == 0 for o in offs):
                ok = False
                break
            x = tuple(d * W[j][t] + sum(o * W[i][t] for i, o in enumerate(offs)) for t in range(n))
            if any(c.denominator != 1 for c in x) or max(abs(c) for c in x) > 4 or nnz(x) < 2:
                ok = False
                break
            X.append(tuple(int(c) for c in x))
        if ok and all(any(x[t] for x in X) for t in range(n)):
            return [tuple(x * (1 if t else 1) for x in X[t]) for t in range(k)]
    raise RuntimeError("그램-슈미트 입력 생성 실패")


def gen_gram_schmidt(rng, i):
    n, k = GS_VARIANTS[i % len(GS_VARIANTS)]
    X = gs_input(rng, n, k)
    xs = mlist([f"{bv('x', j + 1)} = {tup(x)}" for j, x in enumerate(X)])
    st = (
        f"{m(rf'\mathbb{{R}}^{n}')} 의 벡터 {xs} 에 그램-슈미트 과정을 적용하여 "
        f"{m(r'W = \operatorname{span}\{' + ', '.join(bv('x', j + 1) for j in range(k)) + r'\}')} 의 직교기저와 정규직교기저를 구하여라."
    )
    V, steps, _ = gs_steps(X)
    qs = mlist([f"{bv('q', j + 1)} = {unit_tex(w)}" for j, w in enumerate(V)])
    steps.append("<b>정규화:</b> " + qs)
    steps.append(f"검산: 모든 {m(r'i\neq j')} 에 대해 {m(r'\mathbf{v}_i\cdot\mathbf{v}_j = 0')}.")
    ans = (
        f"직교기저: {mlist([f'{bv('v', j + 1)} = {tup(w)}' for j, w in enumerate(V)])}<br>"
        f"정규직교기저: {mlist([f'{bv('q', j + 1)} = {unit_tex(w)}' for j, w in enumerate(V)])}"
        " <span class='note'>(중간 벡터를 상수배한 답도 정답)</span>"
    )
    Vm = sp.Matrix.hstack(*[col(w) for w in V])
    Q = sp.Matrix.hstack(*[col(w) / sp.sqrt(dot(w, w)) for w in V])
    data = dict(X=sp.Matrix.hstack(*[col(x) for x in X]), V=Vm, Q=Q)
    return Problem("gram_schmidt", f"ℝ{SUP[n]}, 벡터 {k}개", st, steps, ans, data)


def gen_qr(rng, i):
    n, k = QR_VARIANTS[i % len(QR_VARIANTS)]
    X = gs_input(rng, n, k)
    A = sp.Matrix.hstack(*[col(x) for x in X])
    st = f"행렬 {m('A = ' + bmat(A))} 의 QR 분해 {m('A = QR')} 를 구하여라. ({m('Q')} 는 정규직교 열을 갖는 {n}×{k} 행렬, {m('R')} 은 대각성분이 양수인 상삼각행렬)"
    steps = [f"{m('A')} 의 열을 {mlist([f'{bv('a', j + 1)} = {tup(x)}' for j, x in enumerate(X)])} 라 하고 그램-슈미트 과정을 적용한다."]
    V, gsteps, _ = gs_steps(X, xs="a")
    steps += gsteps
    Q = sp.Matrix.hstack(*[col(w) / sp.sqrt(dot(w, w)) for w in V])
    R = (Q.T * A).applyfunc(sp.radsimp)
    steps.append("정규화: " + mlist([f"{bv('q', j + 1)} = {unit_tex(w)}" for j, w in enumerate(V)]))
    steps.append(disp(f"Q = {bmat(Q)}"))
    entries = []
    for r_ in range(k):
        for c_ in range(r_, k):
            entries.append(f"r_{{{r_ + 1}{c_ + 1}}} = {bv('q', r_ + 1)}\\cdot{bv('a', c_ + 1)} = {lt(R[r_, c_])}")
    steps.append(
        f"{m('Q^TQ = I')} 이므로 {m('R = Q^TA')} 이고, {m('i>j')} 이면 {m(r'\mathbf{q}_i \perp \mathbf{a}_j')} 이므로 {m('R')} 은 상삼각이다."
        + "<br>" + mlist(entries)
    )
    steps.append(disp(f"R = {bmat(R)}"))
    steps.append(f"검산: {m('QR')} 을 계산하면 {m('A')} 와 같다.")
    ans = disp(f"Q = {bmat(Q)}") + disp(f"R = {bmat(R)}")
    data = dict(A=A, Q=Q, R=R)
    return Problem("qr", f"{n}×{k}", st, steps, ans, data)


# ---------------------------------------------------------------------------
# E. 직교대각화
# ---------------------------------------------------------------------------
DIAG_VARIANTS = ["2", "3r", "3d", "2", "3r", "3d"]


def gen_diag(rng, i):
    var = DIAG_VARIANTS[i % len(DIAG_VARIANTS)]
    A, _ = nice_symmetric(rng, var)
    sol = diag_solution(A)
    hint = " (힌트: 고윳값은 모두 정수이다.)" if A.rows == 3 else ""
    st = f"대칭행렬 {m('A = ' + bmat(A))} 를 직교대각화하여라. 즉 {m('P^TAP = D')} 가 되는 직교행렬 {m('P')} 와 대각행렬 {m('D')} 를 구하여라.{hint}"
    ans = disp(f"P = {bmat(sol['P'])}") + disp(f"D = {bmat(sol['D'])}") + "<span class='note'>(열의 순서·부호, 중근 고유공간의 정규직교기저 선택에 따라 답이 달라질 수 있다)</span>"
    tag = {"2": "2×2", "3d": "3×3 서로 다른 고윳값", "3r": "3×3 중근 (그램-슈미트 필요)"}[var]
    data = dict(A=A, P=sol["P"], D=sol["D"], eig=sol["eig"], need_gs=(var == "3r"))
    return Problem("diag", tag, st, sol["steps"], ans, data)


# ---------------------------------------------------------------------------
# F. 스펙트럼 분해
# ---------------------------------------------------------------------------
SPECTRAL_VARIANTS = ["2", "3r", "2", "3d", "2", "3r"]


def gen_spectral(rng, i):
    var = SPECTRAL_VARIANTS[i % len(SPECTRAL_VARIANTS)]
    power = var == "2"
    A, _ = nice_symmetric(rng, var, exclude=(0,) if power else ())
    sol = diag_solution(A)
    st = f"대칭행렬 {m('A = ' + bmat(A))} 의 스펙트럼 분해 {m(r'A = \sum_i \lambda_i \mathbf{q}_i\mathbf{q}_i^T')} 를 구하여라."
    if power:
        st += f" 또, 이를 이용하여 {m('A^n')} ({m('n')} 은 자연수)을 구하여라."
    steps = ["<b>[1단계] 직교대각화</b>"] + sol["steps"] + ["<b>[2단계] 사영행렬 계산</b>"]
    lams, Ps, ptex, idx = [], [], [], 1
    for lam, vecs in sol["eig"]:
        for w in vecs:
            N = dot(w, w)
            outer = col(w) * col(w).T
            Ps.append(outer / N)
            lams.append(lam)
            ptex.append(f"{bv('q', idx)}{bv('q', idx)}^T = {frac_mat(N, outer)}")
            idx += 1
    steps.append(r"\(\mathbf{q}\mathbf{q}^T = \dfrac{\mathbf{w}\mathbf{w}^T}{\mathbf{w}\cdot\mathbf{w}}\) 를 이용하면" + disp(r"\begin{gathered}" + r" \\ ".join(ptex) + r"\end{gathered}"))
    decomp = signed_comb([(lam, f"{bv('q', j + 1)}{bv('q', j + 1)}^T") for j, lam in enumerate(lams)])
    full = signed_comb([(lam, frac_mat(dot(w, w), col(w) * col(w).T)) for lam, vecs in sol["eig"] for w in vecs])
    steps.append(disp(rf"\begin{{aligned}}A &= {decomp} \\ &= {full}\end{{aligned}}"))
    if var == "3r":
        lr = sol["eig"][[len(v) for _, v in sol["eig"]].index(2)][0]
        steps.append(
            f"<span class='note'>참고: 중근 {m(r'\lambda = ' + lt(lr))} 에 대한 두 항을 합친 {m(r'\mathbf{q}_i\mathbf{q}_i^T + \mathbf{q}_j\mathbf{q}_j^T')} 는 "
            "그 고유공간 위로의 정사영행렬이며, 고유공간의 정규직교기저를 어떻게 골라도 같은 행렬이 된다.</span>"
        )
    ans = disp(rf"A = {full}")
    data = dict(A=A, lams=lams, Ps=Ps)
    if power:
        # 2x2: A^n
        (l1, (w1,)), (l2, (w2,)) = sol["eig"]
        N = dot(w1, w1)
        O1, O2 = col(w1) * col(w1).T, col(w2) * col(w2).T
        An = (sp.Integer(l1) ** NSYM * O1 + sp.Integer(l2) ** NSYM * O2)
        steps.append("<b>[3단계] 거듭제곱</b> " + m(r"\mathbf{q}_i^T\mathbf{q}_j = \delta_{ij}") + " 이므로 " + m(r"(\mathbf{q}_i\mathbf{q}_i^T)^2 = \mathbf{q}_i\mathbf{q}_i^T") + ", 교차항은 0 이 되어")
        steps.append(disp(rf"\begin{{aligned}}A^n &= {lam_paren(l1)}^n\,\mathbf{{q}}_1\mathbf{{q}}_1^T + {lam_paren(l2)}^n\,\mathbf{{q}}_2\mathbf{{q}}_2^T \\ &= {frac_mat(N, An)}\end{{aligned}}"))
        ans += disp(rf"A^n = {frac_mat(N, An)}")
        data["An"] = An / N
    tag = {"2": "2×2 + 거듭제곱", "3d": "3×3", "3r": "3×3 중근"}[var]
    return Problem("spectral", tag, st, steps, ans, data)


# ---------------------------------------------------------------------------
# G. 역문제
# ---------------------------------------------------------------------------
INVERSE_VARIANTS = ["2", "3r", "3d2", "3all", "2", "3r"]


def gen_inverse(rng, i):
    var = INVERSE_VARIANTS[i % len(INVERSE_VARIANTS)]
    kind = {"2": "2", "3r": "3r", "3d2": "3d", "3all": "3d"}[var]
    A, eig = nice_symmetric(rng, kind)
    n = A.rows
    steps = []
    if var == "2":
        (l1, (u1,)), (l2, (u2,)) = eig
        st = f"2×2 대칭행렬 {m('A')} 의 고윳값이 {m(f'{l1},\\ {l2}')} 이고, 고윳값 {m(str(l1))} 에 대응하는 고유벡터가 {m(tup(u1))} 이다. {m('A')} 를 구하여라."
        steps.append(f"대칭행렬의 서로 다른 고윳값에 대응하는 고유벡터는 직교하므로, 고윳값 {m(str(l2))} 의 고유벡터는 {m(tup(u1))} 에 수직인 {m(tup(u2))} 로 잡을 수 있다.")
        pairs = [(l1, u1), (l2, u2)]
        given = [(l1, u1)]
    elif var in ("3d2", "3all"):
        (l1, (u1,)), (l2, (u2,)), (l3, (u3,)) = eig
        if var == "3d2":
            st = (
                f"3×3 대칭행렬 {m('A')} 의 고윳값이 {m(f'{l1},\\ {l2},\\ {l3}')} 이고, 고윳값 {m(str(l1))}, {m(str(l2))} 에 대응하는 고유벡터가 각각 "
                f"{m(tup(u1))}, {m(tup(u2))} 이다. {m('A')} 를 구하여라."
            )
            c = sp.Matrix(u1).cross(sp.Matrix(u2))
            u3p, _ = to_int_primitive(c)
            if dot(u3p, u3) < 0:
                u3p = tuple(-x for x in u3p)
            u3 = u3p
            steps.append(
                f"고윳값 {m(str(l3))} 의 고유벡터는 {m(tup(u1))}, {m(tup(u2))} 모두에 수직이다. 외적을 구하면 {m(tup(c))} 이므로 {m(r'\mathbf{u}_3 = ' + tup(u3))} 로 잡는다."
            )
            given = [(l1, u1), (l2, u2)]
        else:
            st = (
                f"3×3 대칭행렬 {m('A')} 가 고유벡터 {m(r'\mathbf{u}_1 = ' + tup(u1))}, {m(r'\mathbf{u}_2 = ' + tup(u2))}, {m(r'\mathbf{u}_3 = ' + tup(u3))} 를 갖고, "
                f"대응하는 고윳값이 각각 {m(f'{l1},\\ {l2},\\ {l3}')} 이다. {m('A')} 를 구하여라."
            )
            steps.append(f"세 벡터가 서로 직교하는지 확인: {m(r'\mathbf{u}_1\cdot\mathbf{u}_2 = \mathbf{u}_1\cdot\mathbf{u}_3 = \mathbf{u}_2\cdot\mathbf{u}_3 = 0')}.")
            given = [(l1, u1), (l2, u2), (l3, u3)]
        pairs = [(l1, u1), (l2, u2), (l3, u3)]
    else:  # 3r
        (lr, _), (ls, (u,)) = eig
        N = dot(u, u)
        st = (
            f"3×3 대칭행렬 {m('A')} 의 고윳값이 {m(str(lr))} (중근), {m(str(ls))} 이고, 고윳값 {m(str(ls))} 에 대응하는 고유벡터가 {m(tup(u))} 이다. {m('A')} 를 구하여라."
        )
        Pl = col(u) * col(u).T
        steps += [
            f"고윳값 {m(str(lr))} 의 고유공간은 {m(tup(u))} 에 수직인 평면 {m(r'\{\mathbf{x} : ' + tup(u) + r'\cdot\mathbf{x} = 0\}')} 이다 (스펙트럼 정리).",
            f"{m(r'\mathbf{u} = ' + tup(u))} 방향의 정사영행렬은 {m(rf'P_1 = \frac{{\mathbf{{u}}\mathbf{{u}}^T}}{{\mathbf{{u}}\cdot\mathbf{{u}}}} = {frac_mat(N, Pl)}')}, "
            f"평면 위로의 정사영행렬은 {m('P_2 = I - P_1')} 이다.",
            "스펙트럼 분해에 의해" + disp(rf"\begin{{aligned}}A &= {lt(ls)}P_1 + {lam_paren(lr)}P_2 = {lt(lr)}I + ({lt(ls)} - {lam_paren(lr)})P_1 \\ &= {lt(lr)}I + {signed_comb([(sp.Rational(ls - lr, N), bmat(Pl))])} = {bmat(A)}\end{{aligned}}"),
            "<span class='note'>평면의 정규직교기저를 직접 구해 (그램-슈미트) " + m(r"\sum\lambda_i\mathbf{q}_i\mathbf{q}_i^T") + " 로 계산해도 같은 답이 나온다.</span>",
        ]
        ans = m("A = " + bmat(A))
        data = dict(A=A, pairs=[(ls, col(u))], eigvals={lr: 2, ls: 1})
        return Problem("inverse", "3×3 중근", st, steps, ans, data)
    terms = []
    for lam, w in pairs:
        terms.append((sp.Rational(lam, dot(w, w)), bmat(col(w) * col(w).T)))
    steps.append(
        f"{m(r'\mathbf{q}_i = \mathbf{u}_i/\|\mathbf{u}_i\|')} 로 정규화하면 {m(r'A = \sum_i \lambda_i\mathbf{q}_i\mathbf{q}_i^T = \sum_i \frac{\lambda_i}{\mathbf{u}_i\cdot\mathbf{u}_i}\mathbf{u}_i\mathbf{u}_i^T')} 이다."
        + disp(rf"\begin{{aligned}}A &= {signed_comb(terms)} \\ &= {bmat(A)}\end{{aligned}}")
    )
    steps.append(f"검산: 각 {m(r'A\mathbf{u}_i = \lambda_i\mathbf{u}_i')} 가 성립한다.")
    ans = m("A = " + bmat(A))
    ev = {}
    for lam, _ in pairs:
        ev[lam] = ev.get(lam, 0) + 1
    data = dict(A=A, pairs=[(lam, col(w)) for lam, w in pairs], eigvals=ev)
    tag = {"2": "2×2", "3d2": "3×3, 고유벡터 2개만 주어짐", "3all": "3×3"}[var]
    return Problem("inverse", tag, st, steps, ans, data)


# ---------------------------------------------------------------------------
# H. 참/거짓
# ---------------------------------------------------------------------------
def _M(*rows):
    return sp.Matrix(rows)


TF_BANK = [
    dict(s=r"\(W\) 가 \(\mathbb{R}^n\) 의 부분공간이면 \((W^\perp)^\perp = W\) 이다.", a=True,
         e=r"유한차원에서 항상 성립한다. \(W \subseteq (W^\perp)^\perp\) 이고 두 공간의 차원이 모두 \(\dim W\) 로 같다."),
    dict(s=r"서로 직교하는 벡터들의 집합은 항상 일차독립이다.", a=False,
         e=r"영벡터가 포함되면 성립하지 않는다. 반례: \(\{\mathbf{0}, (1,0)\}\) 은 직교집합이지만 일차종속. (영벡터가 아닌 벡터들의 직교집합은 일차독립이다.)",
         c=lambda: sp.Matrix([[0, 1], [0, 0]]).rank() < 2),
    dict(s=r"직교행렬 \(Q\) 의 행렬식은 항상 \(1\) 이다.", a=False,
         e=r"\(\det Q = \pm 1\). 반례: \(Q = \begin{bmatrix}1&0\\0&-1\end{bmatrix}\) 는 \(Q^TQ = I\) 이지만 \(\det Q = -1\).",
         c=lambda: (_M([1, 0], [0, -1]).T * _M([1, 0], [0, -1]) == sp.eye(2)) and _M([1, 0], [0, -1]).det() == -1),
    dict(s=r"실대칭행렬의 서로 다른 고윳값에 대응하는 고유벡터는 서로 직교한다.", a=True,
         e=r"\(\lambda_1\mathbf{v}_1\cdot\mathbf{v}_2 = (A\mathbf{v}_1)\cdot\mathbf{v}_2 = \mathbf{v}_1\cdot(A\mathbf{v}_2) = \lambda_2\mathbf{v}_1\cdot\mathbf{v}_2\) 에서 \(\lambda_1\neq\lambda_2\) 이면 \(\mathbf{v}_1\cdot\mathbf{v}_2 = 0\)."),
    dict(s=r"직교대각화 가능한 실행렬은 대칭행렬이다.", a=True,
         e=r"\(A = PDP^T\) 이면 \(A^T = (PDP^T)^T = PD^TP^T = PDP^T = A\)."),
    dict(s=r"대각화 가능한 실행렬은 모두 직교대각화 가능하다.", a=False,
         e=r"반례: \(A = \begin{bmatrix}1&1\\0&2\end{bmatrix}\) 는 서로 다른 고윳값 \(1, 2\) 를 가져 대각화 가능하지만 대칭이 아니므로 직교대각화 불가능하다. 실제로 고유벡터 \((1,0), (1,1)\) 은 직교하지 않는다.",
         c=lambda: _M([1, 1], [0, 2]).is_diagonalizable() and not _M([1, 1], [0, 2]).is_symmetric()),
    dict(s=r"부분공간 \(W\) 위로의 정사영행렬 \(P\) 는 \(P^2 = P\), \(P^T = P\) 를 만족한다.", a=True,
         e=r"\(P = B(B^TB)^{-1}B^T\) (B 의 열은 W 의 기저)에서 직접 계산하면 두 성질이 나온다. 한 번 사영한 벡터는 다시 사영해도 그대로이다."),
    dict(s=r"\(\{\mathbf{u}_1,\dots,\mathbf{u}_k\}\) 가 \(W\) 의 <b>임의의</b> 기저일 때 \(\operatorname{proj}_W\mathbf{v} = \sum_i \frac{\mathbf{v}\cdot\mathbf{u}_i}{\mathbf{u}_i\cdot\mathbf{u}_i}\mathbf{u}_i\) 이다.", a=False,
         e=r"이 공식은 <b>직교</b>기저일 때만 성립한다. 반례: \(W=\mathbb{R}^2\), \(\mathbf{u}_1=(1,0),\ \mathbf{u}_2=(1,1)\), \(\mathbf{v}=(0,1)\) 이면 공식의 값은 \(0\cdot\mathbf{u}_1 + \frac12\mathbf{u}_2 = (\frac12,\frac12)\) 이지만 실제 \(\operatorname{proj}_W\mathbf{v} = \mathbf{v} = (0,1)\).",
         c=lambda: sp.Rational(0, 1) * _M([1, 0]) + sp.Rational(1, 2) * _M([1, 1]) != _M([0, 1])),
    dict(s=r"\(n\times k\) 행렬 \(Q\) (\(k<n\)) 의 열들이 정규직교이면 \(QQ^T = I_n\) 이다.", a=False,
         e=r"\(Q^TQ = I_k\) 는 성립하지만 \(QQ^T\) 는 \(\operatorname{Col}Q\) 위로의 정사영행렬일 뿐이다. 반례: \(Q = \begin{bmatrix}1\\0\end{bmatrix}\) 이면 \(QQ^T = \begin{bmatrix}1&0\\0&0\end{bmatrix}\neq I\).",
         c=lambda: _M([1], [0]) * _M([1], [0]).T != sp.eye(2)),
    dict(s=r"실대칭행렬의 고윳값은 모두 실수이다.", a=True,
         e=r"스펙트럼 정리의 일부. \(A\mathbf{x}=\lambda\mathbf{x}\) 에서 \(\bar{\mathbf{x}}^TA\mathbf{x}\) 가 실수임을 이용해 \(\lambda=\bar\lambda\) 를 보인다."),
    dict(s=r"실대칭행렬은 중근인 고윳값이 있으면 직교대각화할 수 없다.", a=False,
         e=r"스펙트럼 정리에 의해 모든 실대칭행렬은 직교대각화 가능하다. 중근의 고유공간 차원은 항상 중복도와 같고, 그 안에서 그램-슈미트로 정규직교기저를 고르면 된다. 예: \(\begin{bmatrix}2&1&1\\1&2&1\\1&1&2\end{bmatrix}\) (고윳값 \(4,1,1\)).",
         c=lambda: _M([2, 1, 1], [1, 2, 1], [1, 1, 2]).is_diagonalizable() and _M([2, 1, 1], [1, 2, 1], [1, 1, 2]).eigenvals() == {4: 1, 1: 2}),
    dict(s=r"임의의 행렬 \(A\) 에 대해 \((\operatorname{Row}A)^\perp = \operatorname{Null}A\) 이다.", a=True,
         e=r"\(A\mathbf{x}=\mathbf{0}\) 은 \(\mathbf{x}\) 가 \(A\) 의 모든 행과 내적이 0 이라는 뜻이다. 마찬가지로 \((\operatorname{Col}A)^\perp = \operatorname{Null}A^T\)."),
    dict(s=r"\(W\) 가 \(\mathbb{R}^n\) 의 부분공간이면 \(\dim W + \dim W^\perp = n\) 이다.", a=True,
         e=r"\(W\) 의 기저를 행으로 하는 행렬 \(A\) 에 대해 \(W^\perp = \operatorname{Null}A\) 이고, 차원정리 \(\operatorname{rank}A + \operatorname{nullity}A = n\) 을 쓰면 된다."),
    dict(s=r"두 대칭행렬의 곱은 항상 대칭행렬이다.", a=False,
         e=r"\((AB)^T = B^TA^T = BA\) 이므로 \(AB=BA\) 일 때만 대칭이다. 반례: \(\begin{bmatrix}1&0\\0&0\end{bmatrix}\begin{bmatrix}0&1\\1&0\end{bmatrix} = \begin{bmatrix}0&1\\0&0\end{bmatrix}\).",
         c=lambda: not (_M([1, 0], [0, 0]) * _M([0, 1], [1, 0])).is_symmetric()),
    dict(s=r"대칭행렬 \(A\) 가 가역이면 \(A^{-1}\) 도 대칭행렬이다.", a=True,
         e=r"\((A^{-1})^T = (A^T)^{-1} = A^{-1}\)."),
    dict(s=r"직교행렬 \(Q\) 에 대해 모든 \(\mathbf{x}\) 에서 \(\|Q\mathbf{x}\| = \|\mathbf{x}\|\) 이다.", a=True,
         e=r"\(\|Q\mathbf{x}\|^2 = (Q\mathbf{x})^T(Q\mathbf{x}) = \mathbf{x}^TQ^TQ\mathbf{x} = \mathbf{x}^T\mathbf{x}\)."),
    dict(s=r"정사각행렬의 열들이 서로 직교하면 그 행렬은 직교행렬이다.", a=False,
         e=r"열들이 <b>단위벡터</b>이기도 해야 한다. 반례: \(\begin{bmatrix}1&1\\1&-1\end{bmatrix}\) 의 열은 직교하지만 \(Q^TQ = 2I \neq I\).",
         c=lambda: _M([1, 1], [1, -1]).T * _M([1, 1], [1, -1]) == 2 * sp.eye(2)),
    dict(s=r"\(\mathbf{u}\perp\mathbf{v}\) 이고 \(\mathbf{v}\perp\mathbf{w}\) 이면 \(\mathbf{u}\perp\mathbf{w}\) 이다.", a=False,
         e=r"반례: \(\mathbf{u}=\mathbf{w}=(1,0),\ \mathbf{v}=(0,1)\). \(\mathbf{u}\cdot\mathbf{v}=\mathbf{v}\cdot\mathbf{w}=0\) 이지만 \(\mathbf{u}\cdot\mathbf{w}=1\).",
         c=lambda: dot((1, 0), (0, 1)) == 0 and dot((1, 0), (1, 0)) == 1),
    dict(s=r"\(A\) 의 열들이 일차독립이고 \(A = QR\) 이 QR 분해이면 \(R\) 은 가역이다.", a=True,
         e=r"\(R\) 은 대각성분이 양수(\(r_{ii} = \|\mathbf{v}_i\| > 0\))인 상삼각행렬이므로 \(\det R > 0\)."),
    dict(s=r"임의의 실행렬 \(A\) 에 대해 \(A^TA\) 는 대칭행렬이다.", a=True,
         e=r"\((A^TA)^T = A^T(A^T)^T = A^TA\)."),
    dict(s=r"그램-슈미트 과정에 일차종속인 벡터들을 넣으면 어떤 단계에서 영벡터가 나온다.", a=True,
         e=r"\(\mathbf{x}_k\) 가 앞의 벡터들의 일차결합이면 \(\mathbf{x}_k \in \operatorname{span}\{\mathbf{v}_1,\dots,\mathbf{v}_{k-1}\}\) 이므로 \(\mathbf{v}_k = \mathbf{x}_k - \operatorname{proj}\,\mathbf{x}_k = \mathbf{0}\)."),
    dict(s=r"스펙트럼 분해 \(A=\sum\lambda_i\mathbf{q}_i\mathbf{q}_i^T\) 에서 각 \(\mathbf{q}_i\mathbf{q}_i^T\) 는 계수(rank)가 1 인 정사영행렬이다.", a=True,
         e=r"\(\mathbf{q}_i\mathbf{q}_i^T\mathbf{x} = (\mathbf{q}_i\cdot\mathbf{x})\mathbf{q}_i\) 는 \(\operatorname{span}\{\mathbf{q}_i\}\) 위로의 정사영이다."),
    dict(s=r"\(\mathbf{v}\) 와 \(\operatorname{proj}_W\mathbf{v}\) 사이의 거리는 \(\mathbf{v}\) 와 \(W\) 의 모든 벡터 사이 거리 중 최소이다.", a=True,
         e=r"최량근사정리. \(\mathbf{w}\in W\) 에 대해 \(\|\mathbf{v}-\mathbf{w}\|^2 = \|\mathbf{v}-\operatorname{proj}_W\mathbf{v}\|^2 + \|\operatorname{proj}_W\mathbf{v}-\mathbf{w}\|^2\) (피타고라스)."),
    dict(s=r"\(\mathbb{R}^3\) 에서 평면 \(x+y+z=0\) 의 직교여공간은 직선 \(x=y=z\) 이다.", a=True,
         e=r"평면의 법선벡터 \((1,1,1)\) 이 생성하는 직선이 직교여공간이다.",
         c=lambda: sp.Matrix([[1, 1, 1]]).nullspace() and all(sp.Matrix([1, 1, 1]).dot(b) == 0 for b in sp.Matrix([[1, 1, 1]]).nullspace())),
]


def gen_truefalse(rng, i, order):
    item = TF_BANK[order[i % len(order)]]
    st = "다음 명제의 참/거짓을 판별하고, 거짓이면 반례를 들어라.<br><span class='tf'>" + item["s"] + "</span>"
    ans = "<b>" + ("참" if item["a"] else "거짓") + "</b>"
    return Problem("truefalse", "개념", st, [item["e"]], ans, dict(item=item))


# ---------------------------------------------------------------------------
# 세트 생성
# ---------------------------------------------------------------------------
GENERATORS = dict(
    complement=gen_complement, projection=gen_projection, gram_schmidt=gen_gram_schmidt, qr=gen_qr,
    diag=gen_diag, spectral=gen_spectral, inverse=gen_inverse,
)


def per_kind_counts(count):
    base, rem = divmod(count, len(KINDS))
    return {k: max(1, base + (1 if j < rem else 0)) for j, (k, _, _) in enumerate(KINDS)}


def generate_set(seed, count=48):
    """(문항 목록, 검산 실패로 버린 후보 목록) 을 반환."""
    from verify import verify_problem  # 순환 import 방지

    counts = per_kind_counts(count)
    problems, rejected = [], []
    tf_order = list(range(len(TF_BANK)))
    random.Random(f"{seed}-tf").shuffle(tf_order)
    for kind, _, _ in KINDS:
        seen = set()
        made = 0
        attempt = 0
        while made < counts[kind]:
            rng = random.Random(f"{seed}-{kind}-{made}-{attempt}")
            attempt += 1
            if attempt > 200 * counts[kind]:
                raise RuntimeError(f"{kind} 문항을 충분히 만들지 못함")
            if kind == "truefalse":
                p = gen_truefalse(rng, made + attempt - 1, tf_order)
            else:
                p = GENERATORS[kind](rng, made)
            if p.statement in seen:
                continue
            errs = verify_problem(p)
            if errs:
                rejected.append((p, errs))
                continue
            seen.add(p.statement)
            problems.append(p)
            made += 1
    return problems, rejected


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
SUMMARIES = dict(
    complement=r"\(W^\perp = \{\mathbf{x} : \mathbf{x}\cdot\mathbf{w}=0,\ \forall\mathbf{w}\in W\}\). \(W\) 의 생성벡터를 행으로 하는 \(A\) 에 대해 \(W^\perp = \operatorname{Null}A\), \(\dim W + \dim W^\perp = n\).",
    projection=r"직교기저 \(\{\mathbf{u}_i\}\) 이면 \(\operatorname{proj}_W\mathbf{v} = \sum \frac{\mathbf{v}\cdot\mathbf{u}_i}{\mathbf{u}_i\cdot\mathbf{u}_i}\mathbf{u}_i\). 일반 기저 \(B\) 이면 \(B^TB\hat{\mathbf{x}} = B^T\mathbf{v}\), \(\operatorname{proj}_W\mathbf{v} = B\hat{\mathbf{x}}\). \(\mathbf{v} - \operatorname{proj}_W\mathbf{v} \in W^\perp\).",
    gram_schmidt=r"\(\mathbf{v}_k = \mathbf{x}_k - \sum_{j<k}\frac{\mathbf{x}_k\cdot\mathbf{v}_j}{\mathbf{v}_j\cdot\mathbf{v}_j}\mathbf{v}_j\). 중간 결과를 상수배해도 된다(분수 제거). 마지막에 정규화.",
    qr=r"\(A\) 의 열에 그램-슈미트 → 정규화하여 \(Q\), 그리고 \(R = Q^TA\) (\(r_{ij} = \mathbf{q}_i\cdot\mathbf{a}_j\), 상삼각, 대각 양수).",
    diag=r"고윳값 → 각 고유공간의 기저 → (중근이면 그 고유공간 안에서만 그램-슈미트) → 정규화 → \(P = [\mathbf{q}_1\ \cdots\ \mathbf{q}_n]\), \(P^TAP = D\).",
    spectral=r"\(A = PDP^T = \sum\lambda_i\mathbf{q}_i\mathbf{q}_i^T\), \(\mathbf{q}\mathbf{q}^T = \frac{\mathbf{w}\mathbf{w}^T}{\mathbf{w}\cdot\mathbf{w}}\). 거듭제곱: \(A^n = \sum\lambda_i^n\mathbf{q}_i\mathbf{q}_i^T\).",
    inverse=r"빠진 고유벡터는 직교성으로 찾는다(2차원: 수직 벡터, 3차원: 외적). \(A = \sum\frac{\lambda_i}{\mathbf{u}_i\cdot\mathbf{u}_i}\mathbf{u}_i\mathbf{u}_i^T\). 중근이면 \(A = \lambda_1 I + (\lambda_2-\lambda_1)\frac{\mathbf{u}\mathbf{u}^T}{\mathbf{u}\cdot\mathbf{u}}\).",
    truefalse=r"정의와 정리를 정확히! 거짓인 명제는 가장 간단한 반례를 떠올리는 연습을 하자.",
)

CSS = r"""
:root{--bg:#fbfaf7;--card:#ffffff;--fg:#1d1d1f;--muted:#5f6368;--line:#e3e0d8;--accent:#2357c6;--accent-bg:#eaf0fd;--ans-bg:#eef8f0;--ans-line:#9fd3ab;--sum-bg:#fff7e6;--sum-line:#f0d39a;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#15171a;--card:#1e2125;--fg:#e8e8e6;--muted:#a3a7ad;--line:#33373d;--accent:#8fb1ff;--accent-bg:#22304d;--ans-bg:#1b2d22;--ans-line:#3b6e49;--sum-bg:#2c2617;--sum-line:#6b5a2c;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#15171a;--card:#1e2125;--fg:#e8e8e6;--muted:#a3a7ad;--line:#33373d;--accent:#8fb1ff;--accent-bg:#22304d;--ans-bg:#1b2d22;--ans-line:#3b6e49;--sum-bg:#2c2617;--sum-line:#6b5a2c;color-scheme:dark}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--fg);font-family:"Pretendard","Apple SD Gothic Neo","Noto Sans KR","Malgun Gothic",system-ui,sans-serif;line-height:1.7;font-size:16px}
main{max-width:860px;margin:0 auto;padding:16px 16px 64px}
header h1{font-size:1.6rem;margin:.6em 0 .2em}
header p{color:var(--muted);margin:.2em 0}
nav.sets,nav.toc{display:flex;flex-wrap:wrap;gap:6px;margin:12px 0}
nav a{color:var(--accent);background:var(--accent-bg);text-decoration:none;padding:4px 10px;border-radius:999px;font-size:.9rem}
nav a.cur{background:var(--accent);color:var(--bg)}
.tools{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0}
button{font:inherit;font-size:.9rem;color:var(--fg);background:var(--card);border:1px solid var(--line);border-radius:8px;padding:6px 12px;cursor:pointer}
button:hover{border-color:var(--accent)}
section{margin-top:36px}
section h2{font-size:1.25rem;border-bottom:2px solid var(--accent);padding-bottom:4px}
.summary{background:var(--sum-bg);border:1px solid var(--sum-line);border-radius:10px;padding:10px 14px;font-size:.95rem}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin:14px 0}
.card h3{font-size:1rem;margin:0 0 6px;display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.badge{font-size:.75rem;font-weight:normal;color:var(--muted);border:1px solid var(--line);border-radius:999px;padding:1px 8px}
details{margin-top:10px;border-top:1px dashed var(--line);padding-top:8px}
summary{cursor:pointer;color:var(--accent);font-weight:600}
details.ans>div{background:var(--ans-bg);border-left:4px solid var(--ans-line);padding:8px 12px;border-radius:6px;margin-top:6px}
details.sol ol{padding-left:1.3em;margin:8px 0}
details.sol li{margin:6px 0}
.note{color:var(--muted);font-size:.9em}
.tf{display:block;margin-top:6px;font-weight:600}
mjx-container{max-width:100%}
mjx-container[display="true"]{overflow-x:auto;overflow-y:hidden;padding:2px 0}
mjx-container:not([display]){display:inline-block;max-width:100%;overflow-x:auto;overflow-y:hidden;vertical-align:middle}
footer{margin-top:48px;color:var(--muted);font-size:.85rem}
@media print{.tools,nav{display:none}details{display:block}.card{break-inside:avoid}}
"""

JS = r"""
(function(){
  var root=document.documentElement;
  function get(){try{return localStorage.getItem('la-theme')}catch(e){return null}}
  function set(v){try{localStorage.setItem('la-theme',v)}catch(e){}}
  var t=get(); if(t){root.setAttribute('data-theme',t)}
  document.addEventListener('DOMContentLoaded',function(){
    document.getElementById('theme').addEventListener('click',function(){
      var cur=root.getAttribute('data-theme')||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');
      var nx=cur==='dark'?'light':'dark'; root.setAttribute('data-theme',nx); set(nx);
    });
    function all(sel,open){document.querySelectorAll(sel).forEach(function(d){d.open=open})}
    document.getElementById('openAns').addEventListener('click',function(){all('details.ans',true)});
    document.getElementById('openAll').addEventListener('click',function(){all('details',true)});
    document.getElementById('closeAll').addEventListener('click',function(){all('details',false)});
  });
})();
"""


def render_html(problems, seed, set_no=1, n_sets=1):
    def fname(k):
        return "index.html" if k == 1 else f"set-{k}.html"

    parts = []
    parts.append("<!doctype html>\n<html lang=\"ko\">\n<head>\n<meta charset=\"utf-8\">\n")
    parts.append("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n")
    parts.append(f"<title>선형대수 직교 문제집 {set_no}</title>\n")
    parts.append(f"<style>{CSS}</style>\n<script>{JS}</script>\n")
    parts.append(
        "<script>window.MathJax={tex:{inlineMath:[['\\\\(','\\\\)']],displayMath:[['\\\\[','\\\\]']]},"
        "options:{enableMenu:false},chtml:{scale:1}};</script>\n"
    )
    parts.append(f"<script defer src=\"{MATHJAX_URL}\"></script>\n</head>\n<body>\n<main>\n<header>\n")
    parts.append(f"<h1>선형대수 5장 · 직교 문제집 <small>세트 {set_no}</small></h1>\n")
    parts.append(
        f"<p>직교여공간 · 정사영 · 그램-슈미트 · QR 분해 · 직교대각화 · 스펙트럼 분해. 총 {len(problems)}문항, "
        f"시드 {seed}. 모든 정답은 SymPy 로 정확히 검산되었습니다.</p>\n"
        "<p class=\"note\">먼저 종이에 풀고, ‘정답’으로 확인한 뒤, 막힌 곳만 ‘단계별 풀이’를 펼쳐 보세요.</p>\n"
    )
    if n_sets > 1:
        parts.append("<nav class=\"sets\" aria-label=\"문제집 세트\">" + "".join(
            f"<a href=\"{fname(k)}\"{' class=\"cur\"' if k == set_no else ''}>세트 {k}</a>" for k in range(1, n_sets + 1)) + "</nav>\n")
    parts.append("<nav class=\"toc\" aria-label=\"유형\">" + "".join(
        f"<a href=\"#{k}\">{L}. {t}</a>" for k, L, t in KINDS) + "</nav>\n")
    parts.append(
        "<div class=\"tools\"><button id=\"openAns\" type=\"button\">정답 모두 펼치기</button>"
        "<button id=\"openAll\" type=\"button\">풀이까지 모두 펼치기</button>"
        "<button id=\"closeAll\" type=\"button\">모두 접기</button>"
        "<button id=\"theme\" type=\"button\">라이트/다크</button></div>\n</header>\n"
    )
    num = 0
    for kind, L, title in KINDS:
        ps = [p for p in problems if p.kind == kind]
        parts.append(f"<section id=\"{kind}\">\n<h2>{L}. {title}</h2>\n<div class=\"summary\"><b>핵심</b> {SUMMARIES[kind]}</div>\n")
        for j, p in enumerate(ps):
            num += 1
            pid = f"{L}{j + 1}"
            parts.append(f"<article class=\"card\" id=\"p{pid}\">\n<h3>문제 {num} <span class=\"badge\">{pid} · {p.tag}</span></h3>\n")
            parts.append(f"<div class=\"stmt\">{p.statement}</div>\n")
            parts.append(f"<details class=\"ans\"><summary>정답</summary><div>{p.answer}</div></details>\n")
            parts.append("<details class=\"sol\"><summary>단계별 풀이</summary><ol>\n" + "".join(f"<li>{s}</li>\n" for s in p.steps) + "</ol></details>\n</article>\n")
        parts.append("</section>\n")
    parts.append(
        f"<footer>생성: <code>python generator.py --seed {seed} --count {len(problems)}</code> · 검산: <code>python verify.py</code> · "
        "생성기와 설명은 같은 폴더의 README.md 를 참고하세요.</footer>\n</main>\n</body>\n</html>\n"
    )
    return "".join(parts)


def main(argv=None):
    ap = argparse.ArgumentParser(description="선형대수 5장(직교) 문제집 생성기")
    ap.add_argument("--seed", type=int, default=1, help="난수 시드 (같은 시드 → 같은 문제집)")
    ap.add_argument("--count", type=int, default=48, help="세트당 총 문항 수 (8개 유형에 고르게 배분, 기본 48)")
    ap.add_argument("--sets", type=int, default=1, help="세트 수. 2 이상이면 index.html, set-2.html, … 를 시드 seed, seed+1, … 로 생성")
    ap.add_argument("--out", default=None, help="세트가 1개일 때 출력 파일 (기본: index.html)")
    args = ap.parse_args(argv)
    if args.count < 40:
        print(f"참고: --count {args.count} 이면 유형별 문항이 5개 미만일 수 있습니다.", file=sys.stderr)
    for k in range(1, args.sets + 1):
        seed = args.seed + k - 1
        problems, rejected = generate_set(seed, args.count)
        html = render_html(problems, seed, k, args.sets)
        out = Path(args.out) if (args.out and args.sets == 1) else HERE / ("index.html" if k == 1 else f"set-{k}.html")
        out.write_text(html, encoding="utf-8")
        print(f"[세트 {k}] 시드 {seed}: {len(problems)}문항 → {out.name} (검산 실패로 버린 후보 {len(rejected)}개)")


if __name__ == "__main__":
    main()
