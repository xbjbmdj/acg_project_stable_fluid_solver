#!/usr/bin/env python3
# 读入一个矩阵，判断其是否可逆
"""
Usage examples:
  python keni.py matrix.txt        # whitespace or CSV numbers
  python keni.py matrix.csv --method svd
  python keni.py --example        # create and test with a small example

The script uses NumPy. If NumPy is missing, install with:
  pip install numpy

判定方法：默认使用 SVD 的最小奇异值与容差比较，数值稳定且常用。
"""

from __future__ import annotations
import argparse
import sys
from typing import Tuple


try:
	import numpy as np
except Exception as e:
	print("Error: numpy is required. Install with 'pip install numpy'", file=sys.stderr)
	raise


def load_matrix(path: str):
	if path is None:
		# read from stdin
		data = sys.stdin.read()
		if not data.strip():
			raise SystemExit("No input provided on stdin")
		from io import StringIO

		return np.loadtxt(StringIO(data))

	# try loadtxt first (works with whitespace or comma-separated)
	try:
		return np.loadtxt(path, delimiter=None)
	except Exception:
		try:
			return np.loadtxt(path, delimiter=',')
		except Exception as e:
			raise SystemExit(f"Failed to read matrix from '{path}': {e}")


def svd_invertible(A: np.ndarray) -> Tuple[bool, float, float]:
	"""
	Return (is_invertible, smallest_singular_value, tol)
	tol chosen as max(A.shape) * ||A||_inf * eps
	"""
	if A.ndim != 2 or A.shape[0] != A.shape[1]:
		return False, 0.0, 0.0
	eps = np.finfo(A.dtype).eps
	norm_inf = np.linalg.norm(A, ord=np.inf)
	tol = max(A.shape) * norm_inf * eps
	s = np.linalg.svd(A, compute_uv=False)
	small = float(s[-1])
	return small > tol, small, tol


def det_invertible(A: np.ndarray, tol: float = 1e-12) -> Tuple[bool, float]:
	if A.ndim != 2 or A.shape[0] != A.shape[1]:
		return False, 0.0
	d = float(np.linalg.det(A))
	return abs(d) > tol, d


def lu_invertible(A: np.ndarray, tol: float = None) -> Tuple[bool, float]:
	# Check via LU (using NumPy's LU is not exposed; use Gaussian elimination via scipy or check pivots via LU from numpy.linalg.lu is not available)
	# We'll check via condition number as a proxy for LU pivot problems
	try:
		cond = float(np.linalg.cond(A))
	except Exception:
		return False, float('inf')
	if tol is None:
		tol = 1.0 / np.finfo(A.dtype).eps
	return cond < tol, cond


def main():
	p = argparse.ArgumentParser(description="Read a matrix and test if it's invertible (numeric check).")
	p.add_argument('path', nargs='?', help='Path to matrix file (whitespace or CSV). If omitted, read from stdin.')
	p.add_argument('--method', choices=['svd', 'det', 'cond'], default='svd', help='Method to use (default: svd)')
	p.add_argument('--tol', type=float, default=None, help='Tolerance for determinant or other checks (optional)')
	p.add_argument('--example', action='store_true', help='Run a small built-in example and exit')
	args = p.parse_args()

	if args.example:
		A = np.array([[1.0, 2.0], [3.0, 4.0]])
		print("Example matrix:\n", A)
	else:
		if args.path is None and sys.stdin.isatty():
			p.print_help()
			raise SystemExit(1)
		A = load_matrix(args.path)

	if A.ndim != 2:
		raise SystemExit("Input is not a 2D matrix")

	n, m = A.shape
	print(f"Matrix shape: {n} x {m}")
	if n != m:
		print("Not a square matrix -> not invertible")
		return

	if args.method == 'svd':
		inv, smin, tol = svd_invertible(A)
		print(f"Smallest singular value: {smin:.6g}")
		print(f"Tolerance used: {tol:.6g}")
		print("Invertible (numeric):", "YES" if inv else "NO")
		# show condition number too
		try:
			cond = np.linalg.cond(A)
			print(f"Condition number (2-norm): {cond:.6g}")
		except Exception:
			pass

	elif args.method == 'det':
		tol = args.tol if args.tol is not None else 1e-12
		inv, d = det_invertible(A, tol=tol)
		print(f"Determinant: {d:.6g}")
		print(f"Using tol = {tol}")
		print("Invertible (det):", "YES" if inv else "NO")

	elif args.method == 'cond':
		ok, cond = lu_invertible(A, tol=args.tol)
		print(f"Condition number (approx): {cond:.6g}")
		print("Well-conditioned (cond check):", "YES" if ok else "NO")


if __name__ == '__main__':
	main()

