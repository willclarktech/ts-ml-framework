import { zipWith } from "../utils";
import { mean, sum, Vector } from "./linear";

export interface CostFunction {
	readonly calculate: (expectedInputs: Vector, inputs: Vector) => number;
	readonly derivative: (expectedInputs: Vector, inputs: Vector) => Vector;
}

const crossEntropyCalculate = (expectedInput: number, input: number): number =>
	-(expectedInput * Math.log(input));
const crossEntropyDerivative = (expectedInput: number, input: number): number => -(expectedInput / input);

const crossEntropy: CostFunction = {
	calculate: (expectedInputs: Vector, inputs: Vector) => {
		if (expectedInputs.length !== inputs.length) {
			throw new Error("Cannot calculate cost with inputs/expected inputs of different lengths");
		}
		return sum(zipWith(crossEntropyCalculate, expectedInputs, inputs));
	},
	derivative: (expectedInputs: Vector, inputs: Vector) =>
		zipWith(crossEntropyDerivative, expectedInputs, inputs),
};

const squaredErrorCalculate = (expectedInput: number, input: number): number => (input - expectedInput) ** 2;
const squaredErrorDerivative = (expectedInput: number, input: number): number => 2 * (input - expectedInput);

const meanSquaredError: CostFunction = {
	calculate: (expectedInputs: Vector, inputs: Vector) => {
		if (expectedInputs.length !== inputs.length) {
			throw new Error("Cannot calculate cost with inputs/expected inputs of different lengths");
		}
		return mean(zipWith(squaredErrorCalculate, expectedInputs, inputs));
	},
	derivative: (expectedInputs: Vector, inputs: Vector) =>
		zipWith(squaredErrorDerivative, expectedInputs, inputs),
};

export type CostFunctionName = "cross-entropy" | "mean-squared-error";

export const costFunctionMap = new Map<CostFunctionName, CostFunction>([
	["cross-entropy", crossEntropy],
	["mean-squared-error", meanSquaredError],
]);
