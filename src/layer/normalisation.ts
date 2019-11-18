import {
	NormalisationFunction,
	normalisationFunctionMap,
	NormalisationFunctionName,
	sum,
	transpose,
	Vector,
} from "../maths";
import { zipWith } from "../utils";
import {
	ActivatedLayer,
	ActivationVector,
	BackpropagatedLayer,
	BaseLayer,
	BaseLayerSpecification,
	Delta,
	LayerKind,
} from "./base";

export interface NormalisationLayer extends BaseLayer {
	readonly kind: LayerKind.Normalisation;
	readonly fn: NormalisationFunction;
}

export interface NormalisationLayerSpecification extends BaseLayerSpecification {
	readonly kind: LayerKind.Normalisation;
	readonly fn: NormalisationFunctionName;
}

export const createNormalisationLayer = ({ fn }: NormalisationLayerSpecification): NormalisationLayer => {
	const normalisationFn = normalisationFunctionMap.get(fn);
	if (!normalisationFn) {
		throw new Error("Cannot create normalisation layer");
	}
	return {
		kind: LayerKind.Normalisation,
		fn: normalisationFn,
	};
};

export const activateNormalisationLayer = (
	inputs: ActivationVector,
	layer: NormalisationLayer,
): NormalisationLayer & ActivatedLayer => ({
	...layer,
	inputs,
	activations: layer.fn.calculate(inputs),
});

export const backpropagateNormalisationLayer = (
	layer: NormalisationLayer & ActivatedLayer,
	[subsequentLayer]: readonly BackpropagatedLayer[],
): NormalisationLayer & BackpropagatedLayer => {
	if (!subsequentLayer) {
		throw new Error("Cannot backpropagate normalisation layer without subsequent layer");
	}
	const derivativesMatrix = layer.fn.derivativeInTermsOfOutput
		? layer.fn.derivativeInTermsOfOutput(layer.activations)
		: layer.fn.derivative(layer.inputs);
	const weightedDerivativesMatrix = zipWith(
		(delta: Delta, derivatives: Vector) => derivatives.map(derivative => derivative * delta),
		subsequentLayer.deltas,
		derivativesMatrix,
	);
	return {
		...layer,
		deltas: transpose(weightedDerivativesMatrix).map(sum),
	};
};

export const updateNormalisationLayer = ({
	kind,
	fn,
}: NormalisationLayer & BackpropagatedLayer): NormalisationLayer => ({
	kind,
	fn,
});
