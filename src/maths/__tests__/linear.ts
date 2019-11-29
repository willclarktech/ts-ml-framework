import { add, mean, multiply, shapeOf, subtract, sum, transpose } from "../linear";

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

describe("shapeOf", () => {
	test("works for a number", () => {
		const n = 9;
		expect(shapeOf(n)).toStrictEqual([]);
	});

	test("works for a vector", () => {
		const v = [1, 2, 3];
		expect(shapeOf(v)).toStrictEqual([3]);
	});

	test("works for a matrix", () => {
		const m = [
			[1, 2, 3],
			[4, 5, 6],
		];
		expect(shapeOf(m)).toStrictEqual([2, 3]);
	});

	test("works for a higher-order tensor", () => {
		const m = [
			[
				[1, 2, 3],
				[4, 5, 6],
			],
			[
				[11, 12, 13],
				[14, 15, 16],
			],
			[
				[21, 22, 23],
				[24, 25, 26],
			],
			[
				[31, 32, 33],
				[34, 35, 36],
			],
		];
		expect(shapeOf(m)).toStrictEqual([4, 2, 3]);
	});
});

describe("transpose", () => {
	test("works for a matrix", () => {
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

	test("works for a higher-order tensor", () => {
		const result = transpose([
			[
				[1, 2, 3],
				[4, 5, 6],
				[7, 8, 9],
				[10, 11, 12],
			],
			[
				[101, 102, 103],
				[104, 105, 106],
				[107, 108, 109],
				[110, 111, 112],
			],
			[
				[201, 202, 203],
				[204, 205, 206],
				[207, 208, 209],
				[210, 211, 212],
			],
		]);
		expect(result).toEqual([
			[
				[1, 2, 3],
				[101, 102, 103],
				[201, 202, 203],
			],
			[
				[4, 5, 6],
				[104, 105, 106],
				[204, 205, 206],
			],
			[
				[7, 8, 9],
				[107, 108, 109],
				[207, 208, 209],
			],
			[
				[10, 11, 12],
				[110, 111, 112],
				[210, 211, 212],
			],
		]);
	});
});
