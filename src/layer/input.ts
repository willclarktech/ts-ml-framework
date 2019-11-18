import {
	ActivatedLayer,
	ActivationVectorBatch,
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
	inputsBatch: ActivationVectorBatch,
	layer: InputLayer,
): InputLayer & ActivatedLayer => {
	if (inputsBatch.some(inputs => inputs.length !== layer.width)) {
		throw new Error("Cannot activate input layer with incorrect input width");
	}
	return {
		...layer,
		inputsBatch,
		activationsBatch: inputsBatch,
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
		deltasBatch: subsequentLayer.deltasBatch,
	};
};

export const updateInputLayer = ({ kind, width }: InputLayer & BackpropagatedLayer): InputLayer => ({
	kind,
	width,
});
