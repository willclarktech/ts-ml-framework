import { zipWith } from "../utils";
import { mean, Vector } from "./linear";

export interface CostFunction {
	readonly calculate: (expectedInputs: Vector, inputs: Vector) => number;
	readonly derivative: (expectedInputs: Vector, inputs: Vector) => Vector;
}

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

export type CostFunctionName = "mean-squared-error";

export const costFunctionMap = new Map<CostFunctionName, CostFunction>([
	["mean-squared-error", meanSquaredError],
]);
