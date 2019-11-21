import { ActivationVectorBatch } from "./layer";
import { mean } from "./maths";
import { ActivatedNetwork, activateNetwork, backpropagateNetwork, Network, updateNetwork } from "./network";

export const getAverageError = (network: ActivatedNetwork): number =>
	mean(network.layers[network.layers.length - 1].activationsBatch.map(([error]) => error));

export const trainOnce = (
	expectedOutputs: ActivationVectorBatch,
	inputs: ActivationVectorBatch,
	alpha: number,
	logFrequency: number,
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
) => (network: Network, _: any, i: number) => {
	const activated = activateNetwork(expectedOutputs, inputs, network);
	if (i % logFrequency == 0) {
		const averageTrainError = getAverageError(activated);
		console.info(`${i}: ${averageTrainError}`);
	}
	const backpropagated = backpropagateNetwork(activated);
	return updateNetwork(alpha)(backpropagated);
};

export const train = (
	initialNetwork: Network,
	expectedOutputs: ActivationVectorBatch,
	inputs: ActivationVectorBatch,
	iterations: number,
	alpha = 1,
	logFrequency = 0,
): Network =>
	[...new Array(iterations)].reduce(trainOnce(expectedOutputs, inputs, alpha, logFrequency), initialNetwork);
