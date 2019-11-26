import { Matrix, sum, Vector } from "./linear";
import { kroneckerDelta } from "./utils";

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

const linear: NormalisationFunction = {
	calculate: (inputs: Vector): Vector => {
		const max = Math.max(...inputs.map(Math.abs));
		return inputs.map(i => i / max);
	},
	derivative: (inputs: Vector, really = false): Matrix => {
		if (!really) {
			throw new Error(
				"Linear normalisation should usually be used only once on input. Derivative is extremely slow. Use really flag if you really mean to use this.",
			);
		}
		const max = Math.max(...inputs.map(Math.abs));
		return inputs.map((_, i) => inputs.map((__, j) => kroneckerDelta(i, j) / max));
	},
};

const calculateSoftmax = (inputs: Vector): Vector => {
	const offset = Math.max(...inputs);
	const exponents = inputs.map(input => Math.exp(input - offset));
	const total = sum(exponents);
	return exponents.map(exponent => exponent / total);
};

const softmaxDerivativeInTermsOfOutput = (outputs: Vector): Matrix =>
	outputs.map((output, i) => outputs.map((input, k) => output * (kroneckerDelta(i, k) - input)));

// This is more accurately named "softargmax" but "softmax" is conventional in ML
const softmax: NormalisationFunction = {
	calculate: calculateSoftmax,
	derivative: (inputs: Vector): Matrix => {
		const outputs = calculateSoftmax(inputs);
		return softmaxDerivativeInTermsOfOutput(outputs);
	},
	derivativeInTermsOfOutput: softmaxDerivativeInTermsOfOutput,
};

export type NormalisationFunctionName = "argmax" | "linear" | "softmax";

export const normalisationFunctionMap = new Map<NormalisationFunctionName, NormalisationFunction>([
	["argmax", argmax],
	["linear", linear],
	["softmax", softmax],
]);
