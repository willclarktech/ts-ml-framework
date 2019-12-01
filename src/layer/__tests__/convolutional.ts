import {
	convoluteInputs,
	getConvolutionsForRow,
	take3FromEachRelevantRow,
	take3FromRowAroundIndex,
} from "../convolutional";

describe("take3FromRowAroundIndex", () => {
	const row = [0, 1, 2, 3, 4, 5, 6];

	test("takes 3 from start of row with 0 padding", () => {
		const index = 0;
		expect(take3FromRowAroundIndex(index)(row)).toStrictEqual([0, 0, 1]);
	});

	test("takes 3 from middle of row", () => {
		const index = 3;
		expect(take3FromRowAroundIndex(index)(row)).toStrictEqual([2, 3, 4]);
	});

	test("takes 3 from end of row with 0 padding", () => {
		const index = 6;
		expect(take3FromRowAroundIndex(index)(row)).toStrictEqual([5, 6, 0]);
	});
});

describe("take3FromEachRelevantRow", () => {
	const rows = [
		[0, 1, 2, 3, 4, 5, 6],
		[10, 11, 12, 13, 14, 15, 16],
		[20, 21, 22, 23, 24, 25, 26],
	];

	test("takes 3 from first row with 0 padding", () => {
		const rowIndex = 0;
		const columnIndex = 0;
		expect(take3FromEachRelevantRow(rows, rowIndex)(undefined, columnIndex)).toStrictEqual([
			0,
			0,
			0,
			0,
			0,
			1,
			0,
			10,
			11,
		]);
	});

	test("takes 3 from middle row", () => {
		const rowIndex = 1;
		const columnIndex = 3;
		expect(take3FromEachRelevantRow(rows, rowIndex)(undefined, columnIndex)).toStrictEqual([
			2,
			3,
			4,
			12,
			13,
			14,
			22,
			23,
			24,
		]);
	});

	test("takes 3 from end row with 0 padding", () => {
		const rowIndex = 2;
		const columnIndex = 6;
		expect(take3FromEachRelevantRow(rows, rowIndex)(undefined, columnIndex)).toStrictEqual([
			15,
			16,
			0,
			25,
			26,
			0,
			0,
			0,
			0,
		]);
	});
});

describe("getConvolutionsForRow", () => {
	const rows = [
		[0, 1, 2, 3, 4, 5, 6],
		[10, 11, 12, 13, 14, 15, 16],
		[20, 21, 22, 23, 24, 25, 26],
	];

	test("gets convolutions for a full row", () => {
		const rowIndex = 2;
		expect(getConvolutionsForRow(rows)(rows[rowIndex], rowIndex)).toStrictEqual([
			[0, 10, 11, 0, 20, 21, 0, 0, 0],
			[10, 11, 12, 20, 21, 22, 0, 0, 0],
			[11, 12, 13, 21, 22, 23, 0, 0, 0],
			[12, 13, 14, 22, 23, 24, 0, 0, 0],
			[13, 14, 15, 23, 24, 25, 0, 0, 0],
			[14, 15, 16, 24, 25, 26, 0, 0, 0],
			[15, 16, 0, 25, 26, 0, 0, 0, 0],
		]);
	});
});

describe("convoluteInputs", () => {
	test("works", () => {
		const width = 4;
		const inputs = [10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33];
		expect(convoluteInputs(width)(inputs)).toStrictEqual([
			[0, 0, 0, 0, 10, 11, 0, 20, 21],
			[0, 0, 0, 10, 11, 12, 20, 21, 22],
			[0, 0, 0, 11, 12, 13, 21, 22, 23],
			[0, 0, 0, 12, 13, 0, 22, 23, 0],
			[0, 10, 11, 0, 20, 21, 0, 30, 31],
			[10, 11, 12, 20, 21, 22, 30, 31, 32],
			[11, 12, 13, 21, 22, 23, 31, 32, 33],
			[12, 13, 0, 22, 23, 0, 32, 33, 0],
			[0, 20, 21, 0, 30, 31, 0, 0, 0],
			[20, 21, 22, 30, 31, 32, 0, 0, 0],
			[21, 22, 23, 31, 32, 33, 0, 0, 0],
			[22, 23, 0, 32, 33, 0, 0, 0, 0],
		]);
	});
});
