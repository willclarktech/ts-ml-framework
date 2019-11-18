export const getRandomNumber = (min = -1, max = 1): number => {
	if (max < min) {
		throw new Error("Max cannot be smaller than min");
	}
	const seed = Math.random();
	const range = max - min;
	return seed * range + min;
};

export const zipWith = <T, U, V>(fn: (a: T, b: U) => V, as: readonly T[], bs: readonly U[]): readonly V[] =>
	as.map((a, i) => fn(a, bs[i]));
