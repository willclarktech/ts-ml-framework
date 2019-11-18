import { multiply, sum, transpose, Vector } from "../maths";
import { getRandomNumber, zipWith } from "../utils";
import {
	ActivatedLayer,
	Activation,
	ActivationVector,
	Alpha,
	BackpropagatedLayer,
	BaseLayer,
	BaseLayerSpecification,
	Bias,
	BiasVector,
	Delta,
	DeltaVector,
	LayerKind,
	Weight,
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
	inputs: ActivationVector,
	layer: LinearLayer,
): LinearLayer & ActivatedLayer => ({
	...layer,
	inputs,
	activations: zipWith(
		(ws: WeightVector, bias: Bias) => sum(zipWith(multiply, ws, inputs)) + bias,
		layer.weights,
		layer.biases,
	),
});

export const backpropagateLinearLayer = (
	layer: LinearLayer & ActivatedLayer,
	[subsequentLayer]: readonly BackpropagatedLayer[],
): LinearLayer & BackpropagatedLayer => {
	if (!subsequentLayer) {
		throw new Error("Cannot backpropagate linear layer without subsequent layer");
	}
	const weightedDerivativesMatrix = zipWith(
		(perNodeWeights, delta) => perNodeWeights.map(weight => weight * delta),
		layer.weights,
		subsequentLayer.deltas,
	);
	return {
		...layer,
		deltas: transpose(weightedDerivativesMatrix).map(sum),
	};
};

const updateWeights = (
	alpha: Alpha,
	deltas: Vector,
	weightMatrix: WeightMatrix,
	inputs: ActivationVector,
): WeightMatrix => {
	return zipWith(
		(delta: Delta, weights: WeightVector) =>
			zipWith((weight: Weight, input: Activation) => weight - input * delta * alpha, weights, inputs),
		deltas,
		weightMatrix,
	);
};

const updateBiases = (alpha: Alpha, deltas: Vector, biases: BiasVector): BiasVector => {
	return zipWith((bias: Bias, delta: Delta) => bias - delta * alpha, biases, deltas);
};

export const updateLinearLayer = (
	{ kind, width, weights, biases, inputs }: LinearLayer & BackpropagatedLayer,
	deltas: DeltaVector,
	alpha: Alpha,
): LinearLayer => ({
	kind,
	width,
	weights: updateWeights(alpha, deltas, weights, inputs),
	biases: updateBiases(alpha, deltas, biases),
});
