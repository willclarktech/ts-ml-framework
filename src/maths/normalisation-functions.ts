import { Matrix, sum, Vector } from "./linear";

export interface NormalisationFunction {
	readonly calculate: (inputs: Vector) => Vector;
	readonly derivative: (inputs: Vector) => Matrix;
	readonly derivativeInTermsOfOutput?: (outputs: Vector) => Matrix;
}

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

export type NormalisationFunctionName = "softmax";

export const normalisationFunctionMap = new Map<NormalisationFunctionName, NormalisationFunction>([
	["softmax", softmax],
]);
