#!/usr/bin/env python3
"""
读取若干行，每行包含 A 的一行跟对应的 d 元素（例如：a11 a12 ... a1n b1）
构造方阵 A 和向量 d，求解 Ax = d 并输出解向量 x。

用法示例：
  python qiujie.py matrix.txt
其中 matrix.txt 中每行包含 n+1 个数字（空格或逗号分隔）。

可选参数：
  --example    运行内置示例并退出
  --method     当矩阵奇异或病态时使用的后备方法：solve（默认）/lstsq/pinv

要求：安装 NumPy（`pip install numpy`）。
"""

from __future__ import annotations
import argparse
import sys
from typing import List

try:
	import numpy as np
except Exception:
	print("Error: numpy required. Install with 'pip install numpy'", file=sys.stderr)
	raise


def read_rows(path: str | None) -> np.ndarray:
	data: List[List[float]] = []
	if path is None:
		txt = sys.stdin.read()
		if not txt.strip():
			raise SystemExit('No input provided (stdin empty)')
		lines = txt.strip().splitlines()
	else:
		with open(path, 'r', encoding='utf-8') as f:
			lines = [ln.strip() for ln in f if ln.strip()]

	for i, ln in enumerate(lines, start=1):
		# support comma or whitespace separators
		if ',' in ln and ' ' in ln:
			# prefer comma as separator if both present
			parts = [p for p in ln.replace(',', ' ').split()]
		elif ',' in ln:
			parts = [p for p in ln.split(',') if p.strip()]
		else:
			parts = ln.split()
		try:
			row = [float(x) for x in parts]
		except ValueError as e:
			raise SystemExit(f"Failed to parse numbers on line {i}: {e}")
		if len(row) == 0:
			continue
		data.append(row)

	if not data:
		raise SystemExit('No numeric rows found')

	# ensure consistent row length
	lengths = set(len(r) for r in data)
	if len(lengths) != 1:
		raise SystemExit(f'Inconsistent row lengths: {sorted(lengths)}')

	arr = np.array(data, dtype=float)
	return arr


def solve_system(rows: np.ndarray, method: str = 'solve') -> np.ndarray:
	# rows shape: (n, n+1)
	n, m = rows.shape
	if m != n + 1:
		raise SystemExit(f'Expected each row to have n+1 entries, got shape {rows.shape}')
	A = rows[:, :-1]
	b = rows[:, -1]

	# quick checks
	if A.shape[0] != A.shape[1]:
		raise SystemExit('A is not square')

	try:
		cond = np.linalg.cond(A)
	except Exception:
		cond = float('inf')

	# Try direct solve first if requested
	if method == 'solve':
		try:
			x = np.linalg.solve(A, b)
			return x
		except np.linalg.LinAlgError:
			# fall through to fallback
			pass

	if method in ('lstsq', 'solve'):
		# least-squares (gives a solution even if singular)
		x, resid, rank, s = np.linalg.lstsq(A, b, rcond=None)
		return x

	if method == 'pinv':
		x = np.dot(np.linalg.pinv(A), b)
		return x

	raise SystemExit(f'Unknown method: {method}')


def main():
	p = argparse.ArgumentParser(description='Solve Ax=d from rows where each input row contains A-row then d-element')
	p.add_argument('path', nargs='?', help='Input file path (each line: a1 a2 ... an b). If omitted, read from stdin')
	p.add_argument('--example', action='store_true', help='Run built-in example')
	p.add_argument('--method', choices=['solve', 'lstsq', 'pinv'], default='solve', help='Solution method/fallback')
	args = p.parse_args()

	if args.example:
		# simple example: A = [[2,1],[1,3]], b = [1,2]
		rows = np.array([[2.0, 1.0, 1.0], [1.0, 3.0, 2.0]])
		print('Example rows:')
		print(rows)
		x = solve_system(rows, method=args.method)
		print('Solution x =')
		print(x)
		return

	rows = read_rows(args.path)
	try:
		x = solve_system(rows, method=args.method)
	except SystemExit:
		raise
	except Exception as e:
		raise SystemExit(f'Failed to solve system: {e}')

	# print solution in a compact form
	np.set_printoptions(precision=8, suppress=True)
	print('Solution x:')
	for xi in x:
		print(xi)


if __name__ == '__main__':
	main()

