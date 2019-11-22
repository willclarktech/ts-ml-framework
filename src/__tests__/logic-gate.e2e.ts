import { LayerKind } from "../layer";
import { activateNetwork, createNetwork } from "../network";
import { getAverageError, train } from "../train";
import { nest } from "../utils";

const logFrequency = 0;

test("logical XOR", () => {
	const trainInputs = [
		[0, 0],
		[0, 1],
		[1, 0],
		[1, 1],
	];
	const trainOutputs = nest([0, 1, 1, 0]);
	const specifications = [
		{
			kind: LayerKind.Input as const,
			width: 2,
		},
		{
			kind: LayerKind.Linear as const,
			width: 3,
		},
		{
			kind: LayerKind.NonLinear as const,
			fn: "tanh" as const,
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
	const iterations = 300;
	const alpha = 0.1;
	const trained = train(initialNetwork, trainOutputs, trainInputs, iterations, alpha, logFrequency);
	const tested = activateNetwork(trainOutputs, trainInputs, trained);
	const error = getAverageError(tested);
	expect(error).toBeLessThan(0.01);
});

test("logical XOR, AND", () => {
	const trainInputs = [
		[0, 0],
		[0, 1],
		[1, 0],
		[1, 1],
	];
	const trainOutputs = [
		[0, 0],
		[1, 0],
		[1, 0],
		[0, 1],
	];
	const specifications = [
		{
			kind: LayerKind.Input as const,
			width: 2,
		},
		{
			kind: LayerKind.Linear as const,
			width: 3,
		},
		{
			kind: LayerKind.NonLinear as const,
			fn: "tanh" as const,
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
	const iterations = 300;
	const alpha = 0.1;
	const trained = train(initialNetwork, trainOutputs, trainInputs, iterations, alpha, logFrequency);
	const tested = activateNetwork(trainOutputs, trainInputs, trained);
	const error = getAverageError(tested);
	expect(error).toBeLessThan(0.01);
});
