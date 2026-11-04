#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
문항 정답 검산기 (SymPy 정확 계산).

    python verify.py                      # 시드 1~20 세트를 생성해 모든 문항 검산
    python verify.py --seeds 1-50         # 시드 범위 지정
    python verify.py --check-html         # 저장된 index.html / set-*.html 이 생성기 출력과 같은지도 확인

검산 항목
  A 직교여공간 : W 의 생성벡터 · W⊥ 기저 = 0, W⊥ 기저 일차독립, dim W + dim W⊥ = n
  B 정사영     : proj + perp = v, perp ⟂ W 의 모든 기저, proj ∈ W
  C 그램-슈미트 : V 의 열 서로 직교, 각 k 에 대해 span(x1..xk) = span(v1..vk), Q^TQ = I, Q 열 ∥ V 열
  D QR         : A = QR, Q^TQ = I, R 상삼각 · 대각 양수
  E 직교대각화 : A 대칭, P^TP = I, P^TAP = D (대각), D 의 대각 = 특성다항식의 근(중복도 포함)
  F 스펙트럼   : Σ λ_i P_i = A, P_i^2 = P_i, P_i^T = P_i, P_iP_j = 0, (2×2) A^n 공식 = 실제 A^n (n=1..6)
  G 역문제     : A 대칭, 주어진/유도한 고유쌍에 대해 Au = λu, 고윳값 중복도 일치
  H 참/거짓    : 반례가 있는 문항은 반례를 SymPy 로 재확인
  공통         : 답에 나오는 제곱근은 √2, √3, √5, √6 (및 정수배)만, 분모 ≤ 36
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import sympy as sp

import generator as G

ALLOWED_SQRT = {2, 3, 5, 6}


def _zero(Mx):
    return all(sp.simplify(e) == 0 for e in Mx)


def _eq(A, B):
    return A.shape == B.shape and _zero(A - B)


def _nice(*mats):
    errs = []
    for Mx in mats:
        for e in (Mx if isinstance(Mx, sp.MatrixBase) else [Mx]):
            e = sp.nsimplify(e)
            for p in e.atoms(sp.Pow):
                if p.exp == sp.Rational(1, 2) or p.exp == -sp.Rational(1, 2):
                    if p.base not in ALLOWED_SQRT:
                        errs.append(f"깔끔하지 않은 제곱근 √{p.base}")
            for r in e.atoms(sp.Rational):
                if r.q > 36 or abs(r.p) > 200:
                    errs.append(f"너무 복잡한 수 {r}")
    return errs


def check_complement(d):
    W, perp = d["W_basis"], d["perp"]
    errs = []
    n = W.rows
    B = sp.Matrix.hstack(*perp)
    if not _zero(W.T * B):
        errs.append("W⊥ 기저가 W 와 직교하지 않음")
    if B.rank() != len(perp):
        errs.append("W⊥ 기저가 일차독립이 아님")
    if W.rank() + len(perp) != n:
        errs.append("dim W + dim W⊥ ≠ n")
    return errs + _nice(B, W)


def check_projection(d):
    B, v, proj, perp = d["B"], d["v"], d["proj"], d["perp"]
    errs = []
    if not _eq(proj + perp, v):
        errs.append("proj + perp ≠ v")
    if not _zero(B.T * perp):
        errs.append("perp 가 W 에 수직이 아님")
    if sp.Matrix.hstack(B, proj).rank() != B.rank():
        errs.append("proj 가 W 에 속하지 않음")
    return errs + _nice(proj, perp, sp.sqrt(perp.dot(perp)))


def check_gram_schmidt(d):
    X, V, Q = d["X"], d["V"], d["Q"]
    errs = []
    k = X.cols
    Gm = V.T * V
    if any(Gm[i, j] != 0 for i in range(k) for j in range(k) if i != j) or any(Gm[i, i] == 0 for i in range(k)):
        errs.append("V 의 열이 직교하지 않거나 영벡터 포함")
    for j in range(1, k + 1):
        Xj, Vj = X[:, :j], V[:, :j]
        if not (Xj.rank() == j and Vj.rank() == j and sp.Matrix.hstack(Xj, Vj).rank() == j):
            errs.append(f"span(x1..x{j}) ≠ span(v1..v{j})")
    if not _eq(Q.T * Q, sp.eye(k)):
        errs.append("Q^TQ ≠ I")
    for j in range(k):
        if sp.Matrix.hstack(Q[:, j], V[:, j]).rank() != 1 or sp.simplify(Q[:, j].dot(V[:, j])) <= 0:
            errs.append(f"q{j + 1} 이 v{j + 1} 방향이 아님")
    return errs + _nice(V, Q)


def check_qr(d):
    A, Q, R = d["A"], d["Q"], d["R"]
    errs = []
    k = A.cols
    if not _eq(Q * R, A):
        errs.append("QR ≠ A")
    if not _eq(Q.T * Q, sp.eye(k)):
        errs.append("Q^TQ ≠ I")
    if any(sp.simplify(R[i, j]) != 0 for i in range(k) for j in range(i)):
        errs.append("R 이 상삼각이 아님")
    if any(not (sp.simplify(R[i, i]) > 0) for i in range(k)):
        errs.append("R 의 대각성분이 양수가 아님")
    return errs + _nice(Q, R)


def check_diag(d):
    A, P, D = d["A"], d["P"], d["D"]
    n = A.rows
    errs = []
    if not A.is_symmetric():
        errs.append("A 가 대칭이 아님")
    if not _eq(P.T * P, sp.eye(n)):
        errs.append("P^TP ≠ I")
    PAP = (P.T * A * P).applyfunc(sp.simplify)
    if not _eq(PAP, D) or not D.is_diagonal():
        errs.append("P^TAP ≠ D")
    ev = sp.roots(sp.Poly((A - G.LAM * sp.eye(n)).det(), G.LAM))
    dv = {}
    for i in range(n):
        dv[D[i, i]] = dv.get(D[i, i], 0) + 1
    if ev != dv:
        errs.append("D 의 대각성분이 고윳값과 다름")
    if d.get("need_gs") and max(dv.values()) < 2:
        errs.append("중근 문항인데 중근이 없음")
    return errs + _nice(P)


def check_spectral(d):
    A, lams, Ps = d["A"], d["lams"], d["Ps"]
    n = A.rows
    errs = []
    S = sp.zeros(n)
    for lam, P in zip(lams, Ps):
        S += lam * P
        if not _eq(P * P, P) or not P.is_symmetric():
            errs.append("q q^T 가 정사영행렬이 아님")
    for i in range(len(Ps)):
        for j in range(i):
            if not _zero(Ps[i] * Ps[j]):
                errs.append("P_i P_j ≠ 0")
    if not _eq(S, A):
        errs.append("Σ λ_i q_i q_i^T ≠ A")
    if sum(P.trace() for P in Ps) != n:
        errs.append("사영행렬들의 합이 I 가 아님")
    if "An" in d:
        for k in range(1, 7):
            if not _eq(d["An"].subs(G.NSYM, k), A ** k):
                errs.append(f"A^n 공식이 n={k} 에서 틀림")
                break
    return errs + _nice(*Ps)


def check_inverse(d):
    A = d["A"]
    errs = []
    if not A.is_symmetric():
        errs.append("A 가 대칭이 아님")
    for lam, u in d["pairs"]:
        if not _eq(A * u, lam * u):
            errs.append(f"A u ≠ {lam} u")
    ev = sp.roots(sp.Poly((A - G.LAM * sp.eye(A.rows)).det(), G.LAM))
    if ev != d["eigvals"]:
        errs.append("고윳값/중복도 불일치")
    if any(e.q != 1 for e in A):
        errs.append("A 가 정수행렬이 아님")
    return errs


def check_truefalse(d):
    item = d["item"]
    errs = []
    if not item.get("s") or not item.get("e") or item.get("a") not in (True, False):
        errs.append("문항 형식 오류")
    c = item.get("c")
    if c is not None and not bool(c()):
        errs.append("해설의 반례/예시가 SymPy 검산에 실패")
    return errs


CHECKS = dict(
    complement=check_complement, projection=check_projection, gram_schmidt=check_gram_schmidt, qr=check_qr,
    diag=check_diag, spectral=check_spectral, inverse=check_inverse, truefalse=check_truefalse,
)


def verify_problem(p):
    """오류 메시지 목록을 반환 (빈 목록이면 통과)."""
    try:
        return CHECKS[p.kind](p.data)
    except Exception as e:  # 검산 중 예외도 실패로 취급
        return [f"검산 중 예외: {e!r}"]


def parse_range(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="문제집 SymPy 검산")
    ap.add_argument("--seeds", default="1-20", help="검사할 시드 (예: 1-20 또는 1,5,9)")
    ap.add_argument("--count", type=int, default=48)
    ap.add_argument("--check-html", action="store_true", help="저장된 HTML 이 생성기 출력과 일치하는지 확인 (index.html=시드 1 기준)")
    ap.add_argument("--html-seed", type=int, default=1)
    args = ap.parse_args(argv)

    total = failed = rejected_total = 0
    for seed in parse_range(args.seeds):
        problems, rejected = G.generate_set(seed, args.count)
        rejected_total += len(rejected)
        bad = [(p, verify_problem(p)) for p in problems]
        bad = [(p, e) for p, e in bad if e]
        kinds = {k: sum(1 for p in problems if p.kind == k) for k, _, _ in G.KINDS}
        total += len(problems)
        failed += len(bad)
        status = "통과" if not bad else f"실패 {len(bad)}"
        print(f"시드 {seed:>3}: {len(problems)}문항 {status}  (유형별 {kinds}, 생성 중 버린 후보 {len(rejected)})")
        for p, e in bad:
            print("   ✗", p.kind, p.tag, e)
        for p, e in rejected[:3]:
            print("   (버림)", p.kind, p.tag, e)

    if args.check_html:
        here = Path(__file__).resolve().parent
        pages = sorted(here.glob("set-*.html"), key=lambda q: int(q.stem.split("-")[1]))
        n_sets = 1 + len(pages)
        for k in range(1, n_sets + 1):
            name = "index.html" if k == 1 else f"set-{k}.html"
            seed = args.html_seed + k - 1
            problems, _ = G.generate_set(seed, args.count)
            expected = G.render_html(problems, seed, k, n_sets)
            actual = (here / name).read_text(encoding="utf-8")
            same = expected == actual
            print(f"{name}: {'생성기 출력과 일치' if same else '불일치! 다시 생성하세요'}")
            if not same:
                failed += 1

    print(f"\n총 {total}문항 검산, 실패 {failed}, 생성 단계에서 걸러진 후보 {rejected_total}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
