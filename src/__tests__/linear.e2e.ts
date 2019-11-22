import { LayerKind } from "../layer";
import { activateNetwork, createNetwork } from "../network";
import { getAverageError, train } from "../train";
import { flatten, nest } from "../utils";

const logFrequency = 0;

test("single input, single output: y = 5x + 4", () => {
	const trainInputs = nest([-5, -3, -1, 1, 3, 5]);
	const trainOutputs = nest([-21, -11, -1, 9, 19, 29]);
	const testInputs = nest([-4, -2, 0, 2, 4]);
	const testOutputs = nest([-16, -6, 4, 14, 24]);
	const specifications = [
		{
			kind: LayerKind.Input as const,
			width: 1,
		},
		{
			kind: LayerKind.Linear as const,
			width: 1,
		},
		{
			kind: LayerKind.Cost as const,
			fn: "mean-squared-error" as const,
		},
	];
	const initialNetwork = createNetwork(specifications);
	const iterations = 50;
	const alpha = 0.01;
	const trained = train(initialNetwork, trainOutputs, trainInputs, iterations, alpha, logFrequency);
	const tested = activateNetwork(testOutputs, testInputs, trained);
	const error = getAverageError(tested);
	expect(error).toBeLessThan(0.01);
});

test("single input, multi output: y = [5x + 4, 0.2x - 7]", () => {
	const trainInputs = nest([-5, -3, -1, 1, 3, 5]);
	const trainOutputs = [
		[-21, -8],
		[-11, -7.6],
		[-1, -7.2],
		[9, -6.8],
		[19, -6.4],
		[29, -6],
	];
	const testInputs = nest([-4, -2, 0, 2, 4]);
	const testOutputs = [
		[-16, -7.8],
		[-6, -7.4],
		[4, -7],
		[14, -6.6],
		[24, -6.2],
	];
	const specifications = [
		{
			kind: LayerKind.Input as const,
			width: 1,
		},
		{
			kind: LayerKind.Linear as const,
			width: 2,
		},
		{
			kind: LayerKind.Cost as const,
			fn: "mean-squared-error" as const,
		},
	];
	const initialNetwork = createNetwork(specifications);
	const iterations = 50;
	const alpha = 0.01;
	const trained = train(initialNetwork, trainOutputs, trainInputs, iterations, alpha, logFrequency);
	const tested = activateNetwork(testOutputs, testInputs, trained);
	const error = getAverageError(tested);
	expect(error).toBeLessThan(0.01);
});

test("multi input, single output: z = 5x - 0.2y + 4", () => {
	const trainNs = [-5, -3, -1, 1, 3, 5];
	const trainInputs = flatten(trainNs.map(x => trainNs.map(y => [x, y])));
	const trainOutputs = nest(trainInputs.map(([x, y]) => 5 * x - 0.2 * y + 4));
	const testNs = [-4, -2, 0, 2, 4];
	const testInputs = flatten(testNs.map(x => testNs.map(y => [x, y])));
	const testOutputs = nest(testInputs.map(([x, y]) => 5 * x - 0.2 * y + 4));
	const specifications = [
		{
			kind: LayerKind.Input as const,
			width: 2,
		},
		{
			kind: LayerKind.Linear as const,
			width: 1,
		},
		{
			kind: LayerKind.Cost as const,
			fn: "mean-squared-error" as const,
		},
	];
	const initialNetwork = createNetwork(specifications);
	const iterations = 100;
	const alpha = 0.001;
	const trained = train(initialNetwork, trainOutputs, trainInputs, iterations, alpha, logFrequency);
	const tested = activateNetwork(testOutputs, testInputs, trained);
	const error = getAverageError(tested);
	expect(error).toBeLessThan(0.01);
});
