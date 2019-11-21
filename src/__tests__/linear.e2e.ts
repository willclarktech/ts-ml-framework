import { LayerKind } from "../layer";
import { activateNetwork, createNetwork } from "../network";
import { getAverageError, train } from "../train";
import { nest } from "../utils";

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
