export const getRandomNumber = (min = -1, max = 1): number => {
	if (max < min) {
		throw new Error("Max cannot be smaller than min");
	}
	const seed = Math.random();
	const range = max - min;
	return seed * range + min;
};

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
