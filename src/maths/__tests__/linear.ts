import { add, mean, multiply, subtract, sum, transpose } from "../linear";

const MAX_INACCURACY = 1e-8;

test("add", () => {
	const result = add(2.4, 3.8);
	const diff = Math.abs(result - 6.2);
	expect(diff).toBeLessThan(MAX_INACCURACY);
});

test("subtract", () => {
	const result = subtract(2.4, 3.8);
	const diff = Math.abs(result - -1.4);
	expect(diff).toBeLessThan(MAX_INACCURACY);
});

test("multiply", () => {
	const result = multiply(2.4, 3.8);
	const diff = Math.abs(result - 9.12);
	expect(diff).toBeLessThan(MAX_INACCURACY);
});

test("sum", () => {
	const result = sum([2.4, 3.8, -8.9]);
	const diff = Math.abs(result - -2.7);
	expect(diff).toBeLessThan(MAX_INACCURACY);
});

test("mean", () => {
	const result = mean([2.4, 3.8, -8.9]);
	const diff = Math.abs(result - -0.9);
	expect(diff).toBeLessThan(MAX_INACCURACY);
});

test("transpose", () => {
	const result = transpose([
		[1, 2, 3],
		[4, 5, 6],
		[7, 8, 9],
		[10, 11, 12],
	]);
	expect(result).toEqual([
		[1, 4, 7, 10],
		[2, 5, 8, 11],
		[3, 6, 9, 12],
	]);
});
