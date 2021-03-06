export const createOneHot = (i: number, length = 10): readonly number[] => [
	...new Array(i).fill(0),
	1,
	...new Array(length - 1 - i).fill(0),
];

export const deepMap = <T, U>(
	fn: (item: T, i: number, items: readonly T[]) => U,
	arr: readonly (readonly T[])[],
): readonly (readonly U[])[] => arr.map(a => a.map(fn));

export const getRandomNumber = (min = -1, max = 1): number => {
	if (max < min) {
		throw new Error("Max cannot be smaller than min");
	}
	const seed = Math.random();
	const range = max - min;
	return seed * range + min;
};

export const flatten = <T>(array: readonly (readonly T[])[]): readonly T[] =>
	array.reduce((flattened, next) => [...flattened, ...next], []);

export const nest = (ns: readonly number[]): readonly (readonly number[])[] => ns.map(n => [n]);

export const unzip = <T, U>(zipped: readonly (readonly [T, U])[]): readonly [readonly T[], readonly U[]] =>
	zipped.reduce(
		([as, bs]: readonly [readonly T[], readonly U[]], [a, b]: readonly [T, U]) => [
			[...as, a],
			[...bs, b],
		],
		[[], []],
	);

export const zip = <T, U>(as: readonly T[], bs: readonly U[]): readonly (readonly [T, U])[] =>
	as.map((a, i) => [a, bs[i]]);

export const zipWith = <T, U, V>(fn: (a: T, b: U) => V, as: readonly T[], bs: readonly U[]): readonly V[] =>
	as.map((a, i) => fn(a, bs[i]));

export const zipWith3 = <T, U, V, W>(
	fn: (a: T, b: U, c: V) => W,
	as: readonly T[],
	bs: readonly U[],
	cs: readonly V[],
): readonly W[] => as.map((a, i) => fn(a, bs[i], cs[i]));
