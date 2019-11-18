import {
	ActivatedLayer,
	activateLayer,
	ActivationVector,
	BackpropagatedLayer,
	backpropagateLayer,
	createLayer,
	isCostLayerSpecification,
	isInputLayerSpecification,
	Layer,
	LayerSpecification,
	updateLayer,
} from "./layer";

export interface Network {
	readonly layers: readonly Layer[];
}

export interface ActivatedNetwork extends Network {
	readonly layers: readonly (Layer & ActivatedLayer)[];
}

export interface BackpropagatedNetwork extends Network {
	readonly layers: readonly (Layer & BackpropagatedLayer)[];
}

const createNextLayer = (
	previousLayers: readonly Layer[],
	specification: LayerSpecification,
): readonly Layer[] => {
	const nextLayer = createLayer(specification, previousLayers);
	return [...previousLayers, nextLayer];
};

export const createNetwork = (specifications: readonly LayerSpecification[]): Network => {
	if (!isInputLayerSpecification(specifications[0])) {
		throw new Error("First layer specification must be for an input layer");
	}
	if (!isCostLayerSpecification(specifications[specifications.length - 1])) {
		throw new Error("Last layer specification must be for a cost layer");
	}
	const layers = specifications.reduce(createNextLayer, []);
	return { layers };
};

const activateNextLayer = (expectedOutputs: ActivationVector, inputs: ActivationVector) => (
	previousLayers: readonly (Layer & ActivatedLayer)[],
	nextLayer: Layer,
): readonly (Layer & ActivatedLayer)[] => {
	const outputs = previousLayers.length ? previousLayers[previousLayers.length - 1].activations : inputs;
	const activatedLayer = activateLayer(expectedOutputs, outputs, nextLayer);
	return [...previousLayers, activatedLayer];
};

export const activateNetwork = (
	expectedOutputs: ActivationVector,
	inputs: ActivationVector,
	network: Network,
): ActivatedNetwork => ({
	...network,
	layers: network.layers.reduce(activateNextLayer(expectedOutputs, inputs), []),
});

const backpropagateNextLayer = (
	subsequentLayers: readonly (Layer & BackpropagatedLayer)[],
	nextLayer: Layer & ActivatedLayer,
): readonly (Layer & BackpropagatedLayer)[] => {
	const backpropagatedLayer = backpropagateLayer(nextLayer, subsequentLayers);
	return [backpropagatedLayer, ...subsequentLayers];
};

export const backpropagateNetwork = (network: ActivatedNetwork): BackpropagatedNetwork => ({
	...network,
	layers: [...network.layers].reverse().reduce(backpropagateNextLayer, []),
});

const updateNextLayer = (
	updatedLayers: readonly Layer[],
	nextLayer: Layer & BackpropagatedLayer,
	i: number,
	backpropagatedLayers: readonly (Layer & BackpropagatedLayer)[],
): readonly Layer[] => [...updatedLayers, updateLayer(nextLayer, backpropagatedLayers.slice(i + 1))];

export const updateNetwork = (network: BackpropagatedNetwork): Network => ({
	...network,
	layers: [...network.layers].reduce(updateNextLayer, []),
});
