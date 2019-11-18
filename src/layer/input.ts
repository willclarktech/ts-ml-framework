import {
	ActivatedLayer,
	ActivationVector,
	BackpropagatedLayer,
	BaseLayer,
	BaseLayerSpecification,
	LayerKind,
} from "./base";

export interface InputLayer extends BaseLayer {
	readonly kind: LayerKind.Input;
	readonly width: number;
}

export interface InputLayerSpecification extends BaseLayerSpecification {
	readonly kind: LayerKind.Input;
	readonly width: number;
}

export const createInputLayer = ({ width }: InputLayerSpecification): InputLayer => ({
	kind: LayerKind.Input,
	width,
});

export const activateInputLayer = (
	inputs: ActivationVector,
	layer: InputLayer,
): InputLayer & ActivatedLayer => {
	if (inputs.length !== layer.width) {
		throw new Error("Cannot activate input layer with incorrect input width");
	}
	return {
		...layer,
		inputs,
		activations: inputs,
	};
};

export const backpropagateInputLayer = (
	layer: InputLayer & ActivatedLayer,
	[subsequentLayer]: readonly BackpropagatedLayer[],
): InputLayer & BackpropagatedLayer => {
	if (!subsequentLayer) {
		throw new Error("Cannot backpropagate input layer without subsequent layer");
	}
	return {
		...layer,
		deltas: subsequentLayer.deltas,
	};
};

export const updateInputLayer = ({ kind, width }: InputLayer & BackpropagatedLayer): InputLayer => ({
	kind,
	width,
});
