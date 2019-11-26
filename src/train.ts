import { ActivationVectorBatch } from "./layer";
import { mean } from "./maths";
import {
	ActivatedNetwork,
	activateNetwork,
	activateNetworkWithBatch,
	backpropagateNetwork,
	Batch,
	Network,
	updateNetwork,
} from "./network";
import { zip } from "./utils";

export const getAverageError = (network: ActivatedNetwork): number =>
	mean(network.layers[network.layers.length - 1].activationsBatch.map(([error]) => error));

export const trainOnce = (
	expectedOutputs: ActivationVectorBatch,
	inputs: ActivationVectorBatch,
	alpha: number,
	logFrequency: number,
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
): ((network: Network, _: any, i: number) => Network) => {
	const updateNetworkWithAlpha = updateNetwork(alpha);
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	return (network: Network, _: any, i: number) => {
		const activated = activateNetwork(expectedOutputs, inputs, network);
		if (i % logFrequency == 0) {
			const averageTrainError = getAverageError(activated);
			console.info(`${i}: ${averageTrainError}`);
		}
		const backpropagated = backpropagateNetwork(activated);
		return updateNetworkWithAlpha(backpropagated);
	};
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

export const trainOnceWithBatch = (
	alpha: number,
): ((network: Network, batch: Batch, i: number) => Network) => {
	const updateNetworkWithAlpha = updateNetwork(alpha);
	return (network: Network, batch: Batch): Network => {
		const activated = activateNetworkWithBatch(batch, network);
		const backpropagated = backpropagateNetwork(activated);
		return updateNetworkWithAlpha(backpropagated);
	};
};

const createMiniBatches = (
	expectedOutputs: ActivationVectorBatch,
	inputs: ActivationVectorBatch,
	miniBatchSize: number,
): readonly Batch[] => {
	const nBatches = Math.floor(expectedOutputs.length / miniBatchSize) + 1;
	const zipped = zip(expectedOutputs, inputs);
	return [...new Array(nBatches)].map((_, i) => zipped.slice(i * miniBatchSize, (i + 1) * miniBatchSize));
};

export const trainWithMiniBatches = (
	initialNetwork: Network,
	expectedOutputs: ActivationVectorBatch,
	inputs: ActivationVectorBatch,
	iterations: number,
	alpha = 1,
	logFrequency = 0,
	miniBatchSize = 0,
): Network => {
	if (miniBatchSize < 0) {
		throw new Error("Mini-batch size must be at least 0");
	}
	const miniBatches =
		miniBatchSize === 0
			? [zip(expectedOutputs, inputs)]
			: createMiniBatches(expectedOutputs, inputs, miniBatchSize);
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	return [...new Array(iterations)].reduce((network: Network, _: any, i) => {
		const trained = miniBatches.reduce(trainOnceWithBatch(alpha), network);
		if (i % logFrequency == 0) {
			const activated = activateNetwork(expectedOutputs, inputs, trained);
			const averageTrainError = getAverageError(activated);
			console.info(`${i}: ${averageTrainError}`);
		}
		return trained;
	}, initialNetwork);
};
