export type Vector = readonly number[];
export type Matrix = readonly Vector[];
export type Tensor<T extends number | Vector | Matrix> = readonly (readonly T[])[];

export const add = (a: number, b: number): number => a + b;

export const subtract = (a: number, b: number): number => a - b;

export const multiply = (a: number, b: number): number => a * b;

export const sum = (ns: Vector): number => ns.reduce((total, n) => total + n, 0);

export const mean = (ns: Vector): number => sum(ns) / ns.length;

export const transpose = <T extends number | Vector | Matrix>(matrix: Tensor<T>): Tensor<T> => {
	if (matrix.length === 0 || matrix[0].length === 0) {
		return [];
	}
	return matrix[0].map((_, i: number) => matrix.map(row => row[i]));
};
