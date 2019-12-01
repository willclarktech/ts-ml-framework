import { ActivatedLayer, ActivationVectorBatch, Alpha, BackpropagatedLayer, LayerKind } from "./base";
export {
	ActivatedLayer,
	ActivationVector,
	ActivationVectorBatch,
	Alpha,
	BackpropagatedLayer,
	LayerKind,
} from "./base";
import {
	activateConvolutionalLayer,
	ActivatedConvolutionalLayer,
	backpropagateConvolutionalLayer,
	BackpropagatedConvolutionalLayer,
	ConvolutionalLayer,
	ConvolutionalLayerSpecification,
	createConvolutionalLayer,
	updateConvolutionalLayer,
} from "./convolutional";
import {
	activateCostLayer,
	backpropagateCostLayer,
	CostLayer,
	CostLayerSpecification,
	createCostLayer,
	updateCostLayer,
} from "./cost";
import {
	activateInputLayer,
	backpropagateInputLayer,
	createInputLayer,
	InputLayer,
	InputLayerSpecification,
	updateInputLayer,
} from "./input";
import {
	activateLinearLayer,
	backpropagateLinearLayer,
	createLinearLayer,
	LinearLayer,
	LinearLayerSpecification,
	updateLinearLayer,
} from "./linear";
import {
	activateNonLinearLayer,
	backpropagateNonLinearLayer,
	createNonLinearLayer,
	NonLinearLayer,
	NonLinearLayerSpecification,
	updateNonLinearLayer,
} from "./non-linear";
import {
	activateNormalisationLayer,
	backpropagateNormalisationLayer,
	createNormalisationLayer,
	NormalisationLayer,
	NormalisationLayerSpecification,
	updateNormalisationLayer,
} from "./normalisation";

export type LayerSpecification =
	| InputLayerSpecification
	| ConvolutionalLayerSpecification
	| LinearLayerSpecification
	| NonLinearLayerSpecification
	| NormalisationLayerSpecification
	| CostLayerSpecification;

export const isInputLayerSpecification = (
	specification: LayerSpecification,
): specification is InputLayerSpecification => {
	return specification.kind === LayerKind.Input;
};

export const isCostLayerSpecification = (
	specification: LayerSpecification,
): specification is CostLayerSpecification => {
	return specification.kind === LayerKind.Cost;
};

export type Layer =
	| InputLayer
	| ConvolutionalLayer
	| LinearLayer
	| NonLinearLayer
	| NormalisationLayer
	| CostLayer;

export const isInputLayer = (layer: Layer): layer is InputLayer => {
	return layer.kind === LayerKind.Input;
};

export const isLinearLayer = (layer: Layer): layer is LinearLayer => {
	return layer.kind === LayerKind.Linear;
};

const getOutputWidth = (layers: readonly Layer[]): number =>
	layers.reduce(
		(lastWidth: number, layer: Layer) =>
			isInputLayer(layer) || isLinearLayer(layer) ? layer.width : lastWidth,
		-1,
	);

export const createLayer = (specification: LayerSpecification, previousLayers: readonly Layer[]): Layer => {
	switch (specification.kind) {
		case LayerKind.Input:
			return createInputLayer(specification);
		case LayerKind.Convolutional: {
			const previousWidth = getOutputWidth(previousLayers);
			return createConvolutionalLayer(specification, previousWidth);
		}
		case LayerKind.Linear: {
			const previousWidth = getOutputWidth(previousLayers);
			return createLinearLayer(specification, previousWidth);
		}
		case LayerKind.NonLinear:
			return createNonLinearLayer(specification);
		case LayerKind.Normalisation:
			return createNormalisationLayer(specification);
		case LayerKind.Cost:
			return createCostLayer(specification);
	}
	throw new Error("Cannot create layer for specification");
};

export const activateLayer = (
	expectedOutputs: ActivationVectorBatch,
	inputs: ActivationVectorBatch,
	layer: Layer,
): Layer & ActivatedLayer => {
	switch (layer.kind) {
		case LayerKind.Input:
			return activateInputLayer(inputs, layer);
		case LayerKind.Convolutional:
			return activateConvolutionalLayer(inputs, layer);
		case LayerKind.Linear:
			return activateLinearLayer(inputs, layer);
		case LayerKind.NonLinear:
			return activateNonLinearLayer(inputs, layer);
		case LayerKind.Normalisation:
			return activateNormalisationLayer(inputs, layer);
		case LayerKind.Cost:
			return activateCostLayer(expectedOutputs, inputs, layer);
	}
	throw new Error("Cannot activate layer of unknown kind");
};

export const backpropagateLayer = (
	layer: Layer & ActivatedLayer,
	subsequentLayers: readonly (Layer & BackpropagatedLayer)[],
): Layer & BackpropagatedLayer => {
	switch (layer.kind) {
		case LayerKind.Input:
			return backpropagateInputLayer(layer, subsequentLayers);
		case LayerKind.Convolutional:
			return backpropagateConvolutionalLayer(
				layer as ConvolutionalLayer & ActivatedConvolutionalLayer & ActivatedLayer,
				subsequentLayers,
			);
		case LayerKind.Linear:
			return backpropagateLinearLayer(layer, subsequentLayers);
		case LayerKind.NonLinear:
			return backpropagateNonLinearLayer(layer, subsequentLayers);
		case LayerKind.Normalisation:
			return backpropagateNormalisationLayer(layer, subsequentLayers);
		case LayerKind.Cost:
			return backpropagateCostLayer(layer, subsequentLayers);
	}
	throw new Error("Cannot backpropagate layer of unknown kind");
};

export const updateLayer = (alpha: Alpha) => (
	layer: Layer & BackpropagatedLayer,
	subsequentLayers: readonly (Layer & BackpropagatedLayer)[],
): Layer => {
	switch (layer.kind) {
		case LayerKind.Input:
			return updateInputLayer(layer);
		case LayerKind.Convolutional: {
			const nextLayerDeltas = subsequentLayers[0].deltasBatch;
			return updateConvolutionalLayer(
				layer as ConvolutionalLayer & BackpropagatedConvolutionalLayer & BackpropagatedLayer,
				nextLayerDeltas,
				alpha,
			);
		}
		case LayerKind.Linear: {
			const nextLayerDeltas = subsequentLayers[0].deltasBatch;
			return updateLinearLayer(layer, nextLayerDeltas, alpha);
		}
		case LayerKind.NonLinear:
			return updateNonLinearLayer(layer);
		case LayerKind.Normalisation:
			return updateNormalisationLayer(layer);
		case LayerKind.Cost:
			return updateCostLayer(layer);
	}
	throw new Error("Cannot update layer of unknown kind");
};
