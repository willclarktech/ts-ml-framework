import { multiply, NonLinearFunction, nonLinearFunctionMap, NonLinearFunctionName, Vector } from "../maths";
import { zipWith } from "../utils";
import {
	ActivatedLayer,
	ActivationVectorBatch,
	BackpropagatedLayer,
	BaseLayer,
	BaseLayerSpecification,
	DeltaVector,
	LayerKind,
} from "./base";

export interface NonLinearLayer extends BaseLayer {
	readonly kind: LayerKind.NonLinear;
	readonly fn: NonLinearFunction;
}

export interface NonLinearLayerSpecification extends BaseLayerSpecification {
	readonly kind: LayerKind.NonLinear;
	readonly fn: NonLinearFunctionName;
}

export const createNonLinearLayer = ({ fn }: NonLinearLayerSpecification): NonLinearLayer => {
	const nonLinearFn = nonLinearFunctionMap.get(fn);
	if (!nonLinearFn) {
		throw new Error("Cannot create non-linear layer");
	}
	return {
		kind: LayerKind.NonLinear,
		fn: nonLinearFn,
	};
};

export const activateNonLinearLayer = (
	inputsBatch: ActivationVectorBatch,
	layer: NonLinearLayer,
): NonLinearLayer & ActivatedLayer => ({
	...layer,
	inputsBatch,
	activationsBatch: inputsBatch.map(inputs => inputs.map(layer.fn.calculate)),
});

export const backpropagateNonLinearLayer = (
	layer: NonLinearLayer & ActivatedLayer,
	[subsequentLayer]: readonly BackpropagatedLayer[],
): NonLinearLayer & BackpropagatedLayer => {
	if (!subsequentLayer) {
		throw new Error("Cannot backpropagate non-linear layer without subsequent layer");
	}
	const { derivativeInTermsOfOutput } = layer.fn;
	const derivativesBatch = derivativeInTermsOfOutput
		? layer.activationsBatch.map(activations => activations.map(derivativeInTermsOfOutput))
		: layer.inputsBatch.map(inputs => inputs.map(layer.fn.derivative));
	return {
		...layer,
		deltasBatch: zipWith(
			(deltas: DeltaVector, derivatives: Vector) => zipWith(multiply, deltas, derivatives),
			subsequentLayer.deltasBatch,
			derivativesBatch,
		),
	};
};

export const updateNonLinearLayer = ({ kind, fn }: NonLinearLayer & BackpropagatedLayer): NonLinearLayer => ({
	kind,
	fn,
});
