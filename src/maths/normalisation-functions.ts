import { Matrix, sum, Vector } from "./linear";

export interface NormalisationFunction {
	readonly calculate: (inputs: Vector) => Vector;
	readonly derivative: (inputs: Vector) => Matrix;
	readonly derivativeInTermsOfOutput?: (outputs: Vector) => Matrix;
}

const argmax: NormalisationFunction = {
	calculate: (inputs: Vector): Vector => {
		const max = Math.max(...inputs);
		const nMax = inputs.filter(n => n === max).length;
		const value = 1 / nMax;
		return inputs.map(n => (n === max ? value : 0));
	},
	derivative: (_inputs: Vector) => {
		throw new Error("Cannot backpropagate argmax");
	},
};

const calculateSoftmax = (inputs: Vector): Vector => {
	const offset = Math.max(...inputs);
	const exponents = inputs.map(input => Math.exp(input - offset));
	const total = sum(exponents);
	return exponents.map(exponent => exponent / total);
};

const softmaxDerivativeInTermsOfOutput = (outputs: Vector): Matrix =>
	outputs.map((output, i) => outputs.map((input, k) => output * ((i === k ? 1 : 0) - input)));

// This is more accurately named "softargmax" but "softmax" is conventional in ML
const softmax: NormalisationFunction = {
	calculate: calculateSoftmax,
	derivative: (inputs: Vector) => {
		const outputs = calculateSoftmax(inputs);
		return softmaxDerivativeInTermsOfOutput(outputs);
	},
	derivativeInTermsOfOutput: softmaxDerivativeInTermsOfOutput,
};

export type NormalisationFunctionName = "argmax" | "softmax";

export const normalisationFunctionMap = new Map<NormalisationFunctionName, NormalisationFunction>([
	["argmax", argmax],
	["softmax", softmax],
]);
