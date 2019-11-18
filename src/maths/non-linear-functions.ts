export interface NonLinearFunction {
	readonly calculate: (input: number) => number;
	readonly derivative: (input: number) => number;
	readonly derivativeInTermsOfOutput?: (output: number) => number;
}

const relu: NonLinearFunction = {
	calculate: (input: number) => Math.max(0, input),
	derivative: (input: number) => Number(input > 0),
	derivativeInTermsOfOutput: (output: number) => Number(output > 0),
};

const calculateSigmoid = (input: number): number => 1 / (1 + Math.exp(-input));

const sigmoid: NonLinearFunction = {
	calculate: calculateSigmoid,
	derivative: (input: number) => {
		const s = calculateSigmoid(input);
		return s * (1 - s);
	},
	derivativeInTermsOfOutput: (output: number) => output * (1 - output),
};

const tanh: NonLinearFunction = {
	calculate: Math.tanh,
	derivative: (input: number) => 1 - Math.tanh(input) ** 2,
	derivativeInTermsOfOutput: (output: number) => 1 - output ** 2,
};

export type NonLinearFunctionName = "relu" | "sigmoid" | "tanh";

export const nonLinearFunctionMap = new Map<NonLinearFunctionName, NonLinearFunction>([
	["relu", relu],
	["sigmoid", sigmoid],
	["tanh", tanh],
]);
