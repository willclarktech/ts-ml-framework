import { LayerKind } from "../layer";
import { mean } from "../maths";
import {
	ActivatedNetwork,
	activateNetwork,
	backpropagateNetwork,
	createNetwork,
	Network,
	updateNetwork,
} from "../network";

const logFrequeney = 1;

const nest = (ns: readonly number[]): readonly (readonly number[])[] => ns.map(n => [n]);

const getAverageError = (network: ActivatedNetwork): number =>
	mean(network.layers[network.layers.length - 1].activationsBatch.map(([error]) => error));

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
	const iterations = [...new Array(50)];
	const alpha = 0.01;
	const trained = iterations.reduce((network: Network, _, i: number) => {
		const activated = activateNetwork(trainOutputs, trainInputs, network);
		const averageTrainError = getAverageError(activated);
		if (i % logFrequeney === 0) {
			console.info(`i: ${i}; err: ${averageTrainError}`);
		}
		const backpropagated = backpropagateNetwork(activated);
		return updateNetwork(alpha)(backpropagated);
	}, initialNetwork);
	const tested = activateNetwork(testOutputs, testInputs, trained);
	const error = getAverageError(tested);
	expect(error).toBeLessThan(0.01);
});
