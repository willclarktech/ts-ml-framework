import { multiply, NonLinearFunction, nonLinearFunctionMap, NonLinearFunctionName } from "../maths";
import { zipWith } from "../utils";
import {
	ActivatedLayer,
	ActivationVector,
	BackpropagatedLayer,
	BaseLayer,
	BaseLayerSpecification,
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
	inputs: ActivationVector,
	layer: NonLinearLayer,
): NonLinearLayer & ActivatedLayer => ({
	...layer,
	inputs,
	activations: inputs.map(layer.fn.calculate),
});

export const backpropagateNonLinearLayer = (
	layer: NonLinearLayer & ActivatedLayer,
	[subsequentLayer]: readonly BackpropagatedLayer[],
): NonLinearLayer & BackpropagatedLayer => {
	if (!subsequentLayer) {
		throw new Error("Cannot backpropagate non-linear layer without subsequent layer");
	}
	const derivatives = layer.fn.derivativeInTermsOfOutput
		? layer.activations.map(layer.fn.derivativeInTermsOfOutput)
		: layer.inputs.map(layer.fn.derivative);
	return {
		...layer,
		deltas: zipWith(multiply, subsequentLayer.deltas, derivatives),
	};
};
