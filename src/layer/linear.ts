import { Matrix, multiply, subtract, sum, transpose, Vector } from "../maths";
import { getRandomNumber, zip, zipWith } from "../utils";
import {
	ActivatedLayer,
	Activation,
	ActivationVectorBatch,
	Alpha,
	BackpropagatedLayer,
	BaseLayer,
	BaseLayerSpecification,
	Bias,
	BiasVector,
	Delta,
	DeltaVectorBatch,
	LayerKind,
	WeightMatrix,
	WeightVector,
} from "./base";

export interface LinearLayer extends BaseLayer {
	readonly kind: LayerKind.Linear;
	readonly width: number;
	readonly weights: WeightMatrix;
	readonly biases: BiasVector;
}

export interface LinearLayerSpecification extends BaseLayerSpecification {
	readonly kind: LayerKind.Linear;
	readonly width: number;
}

export const createLinearLayer = (
	{ width }: LinearLayerSpecification,
	previousWidth: number,
): LinearLayer => {
	if (previousWidth < 0) {
		throw new Error("Invalid previous layer width");
	}
	const weights = [...new Array(width)].map(_ => [...new Array(previousWidth)].map(__ => getRandomNumber()));
	const biases = [...new Array(width)].map(_ => getRandomNumber());
	return {
		kind: LayerKind.Linear,
		width,
		weights,
		biases,
	};
};

export const activateLinearLayer = (
	inputsBatch: ActivationVectorBatch,
	layer: LinearLayer,
): LinearLayer & ActivatedLayer => ({
	...layer,
	inputsBatch,
	activationsBatch: inputsBatch.map(inputs =>
		zipWith(
			(ws: WeightVector, bias: Bias) => sum(zipWith(multiply, ws, inputs)) + bias,
			layer.weights,
			layer.biases,
		),
	),
});

export const backpropagateLinearLayer = (
	layer: LinearLayer & ActivatedLayer,
	[subsequentLayer]: readonly BackpropagatedLayer[],
): LinearLayer & BackpropagatedLayer => {
	if (!subsequentLayer) {
		throw new Error("Cannot backpropagate linear layer without subsequent layer");
	}
	const weightedDerivativesMatrices = subsequentLayer.deltasBatch.map(deltas =>
		zipWith((perNodeWeights, delta) => perNodeWeights.map(weight => weight * delta), layer.weights, deltas),
	);
	return {
		...layer,
		deltasBatch: weightedDerivativesMatrices.map(weightedDerivativesMatrix =>
			transpose(weightedDerivativesMatrix).map(sum),
		),
	};
};

const updateWeightsForBatch = (
	alpha: Alpha,
	deltasBatch: DeltaVectorBatch,
	weightMatrix: WeightMatrix,
	inputsBatch: ActivationVectorBatch,
): WeightMatrix => {
	const nullWeightUpdates = [...new Array(weightMatrix.length)].map(() =>
		new Array(weightMatrix[0].length).fill(0),
	);
	const weightUpdates: readonly (readonly number[])[] = zip(deltasBatch, inputsBatch).reduce(
		(updatesMatrix: Matrix, [deltas, inputs]) =>
			zipWith(
				(delta: Delta, updates: Vector) =>
					zipWith((update: number, input: Activation) => update + input * delta * alpha, updates, inputs),
				deltas,
				updatesMatrix,
			),
		nullWeightUpdates,
	);
	return zipWith(
		(weights, updates) => zipWith((weight, update) => weight - update, weights, updates),
		weightMatrix,
		weightUpdates,
	);
};

const updateBiasesForBatch = (
	alpha: Alpha,
	deltasBatch: DeltaVectorBatch,
	biases: BiasVector,
): BiasVector => {
	const nullBiasUpdates = new Array(biases.length).fill(0);
	const biasUpdates: readonly number[] = deltasBatch.reduce(
		(updates: readonly number[], deltas: readonly number[]) =>
			zipWith((update, delta) => update + delta * alpha, updates, deltas),
		nullBiasUpdates,
	);
	return zipWith(subtract, biases, biasUpdates);
};

export const updateLinearLayer = (
	{ kind, width, weights, biases, inputsBatch }: LinearLayer & BackpropagatedLayer,
	deltasBatch: DeltaVectorBatch,
	alpha: Alpha,
): LinearLayer => ({
	kind,
	width,
	weights: updateWeightsForBatch(alpha, deltasBatch, weights, inputsBatch),
	biases: updateBiasesForBatch(alpha, deltasBatch, biases),
});
