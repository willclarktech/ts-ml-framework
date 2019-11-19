import { nonLinearFunctionMap } from "../non-linear-functions";

const MAX_INACCURACY = 1e-8;

describe("relu", () => {
	const relu = nonLinearFunctionMap.get("relu");
	if (!relu) {
		throw new Error("relu not found");
	}

	test("calculate", () => {
		const cases = [
			[0, 0],
			[1, 1],
			[-1, 0],
			[10, 10],
			[-10, 0],
		];
		cases.forEach(([input, output]) => {
			const result = relu.calculate(input);
			const diff = Math.abs(result - output);
			expect(diff).toBeLessThan(MAX_INACCURACY);
		});
	});

	test("derivative", () => {
		const cases = [
			[0, 0],
			[1, 1],
			[-1, 0],
			[10, 1],
			[-10, 0],
		];
		cases.forEach(([input, output]) => {
			const result = relu.derivative(input);
			const diff = Math.abs(result - output);
			expect(diff).toBeLessThan(MAX_INACCURACY);
		});
	});

	test("derivativeInTermsOfOutput", () => {
		const { derivativeInTermsOfOutput } = relu;
		if (!derivativeInTermsOfOutput) {
			throw new Error("relu.derivativeInTermsOfOutput not defined");
		}
		const cases = [
			[0, 0],
			[1, 1],
			[10, 1],
		];
		cases.forEach(([input, output]) => {
			const result = derivativeInTermsOfOutput(input);
			const diff = Math.abs(result - output);
			expect(diff).toBeLessThan(MAX_INACCURACY);
		});
	});
});

describe("sigmoid", () => {
	const sigmoid = nonLinearFunctionMap.get("sigmoid");
	if (!sigmoid) {
		throw new Error("sigmoid not found");
	}

	test("calculate", () => {
		const cases = [
			[0, 0.5],
			[1, 0.73105857863000487],
			[-1, 0.26894142136999512],
		];
		cases.forEach(([input, output]) => {
			const result = sigmoid.calculate(input);
			const diff = Math.abs(result - output);
			expect(diff).toBeLessThan(MAX_INACCURACY);
		});
	});

	test("derivative", () => {
		const cases = [
			[0, 0.25],
			[1, 0.19661193324148185],
			[-1, 0.19661193324148185],
		];
		cases.forEach(([input, output]) => {
			const result = sigmoid.derivative(input);
			const diff = Math.abs(result - output);
			expect(diff).toBeLessThan(MAX_INACCURACY);
		});
	});

	test("derivativeInTermsOfOutput", () => {
		const { derivativeInTermsOfOutput } = sigmoid;
		if (!derivativeInTermsOfOutput) {
			throw new Error("sigmoid.derivativeInTermsOfOutput not defined");
		}
		const cases = [
			[0.5, 0.25],
			[0.73105857863000487, 0.19661193324148185],
			[0.26894142136999512, 0.19661193324148185],
		];
		cases.forEach(([input, output]) => {
			const result = derivativeInTermsOfOutput(input);
			const diff = Math.abs(result - output);
			expect(diff).toBeLessThan(MAX_INACCURACY);
		});
	});
});

describe("tanh", () => {
	const tanh = nonLinearFunctionMap.get("tanh");
	if (!tanh) {
		throw new Error("tanh not found");
	}

	test("calculate", () => {
		const cases = [
			[0, 0],
			[1, 0.761594155956],
			[-1, -0.761594155956],
		];
		cases.forEach(([input, output]) => {
			const result = tanh.calculate(input);
			const diff = Math.abs(result - output);
			expect(diff).toBeLessThan(MAX_INACCURACY);
		});
	});

	test("derivative", () => {
		const cases = [
			[0, 1],
			[1, 0.41997434161402614],
			[-1, 0.41997434161402614],
		];
		cases.forEach(([input, output]) => {
			const result = tanh.derivative(input);
			const diff = Math.abs(result - output);
			expect(diff).toBeLessThan(MAX_INACCURACY);
		});
	});

	test("derivativeInTermsOfOutput", () => {
		const { derivativeInTermsOfOutput } = tanh;
		if (!derivativeInTermsOfOutput) {
			throw new Error("tanh.derivativeInTermsOfOutput not defined");
		}
		const cases = [
			[0, 1],
			[0.761594155956, 0.41997434161402614],
			[-0.761594155956, 0.41997434161402614],
		];
		cases.forEach(([input, output]) => {
			const result = derivativeInTermsOfOutput(input);
			const diff = Math.abs(result - output);
			expect(diff).toBeLessThan(MAX_INACCURACY);
		});
	});
});
