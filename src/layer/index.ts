import { ActivatedLayer, ActivationVector, BackpropagatedLayer, LayerKind } from "./base";
export { ActivatedLayer, Activation, ActivationVector, BackpropagatedLayer, LayerKind } from "./base";
import {
	activateCostLayer,
	backpropagateCostLayer,
	CostLayer,
	CostLayerSpecification,
	createCostLayer,
} from "./cost";
import {
	activateInputLayer,
	backpropagateInputLayer,
	createInputLayer,
	InputLayer,
	InputLayerSpecification,
} from "./input";
import {
	activateLinearLayer,
	backpropagateLinearLayer,
	createLinearLayer,
	LinearLayer,
	LinearLayerSpecification,
} from "./linear";
import {
	activateNonLinearLayer,
	backpropagateNonLinearLayer,
	createNonLinearLayer,
	NonLinearLayer,
	NonLinearLayerSpecification,
} from "./non-linear";
import {
	activateNormalisationLayer,
	backpropagateNormalisationLayer,
	createNormalisationLayer,
	NormalisationLayer,
	NormalisationLayerSpecification,
} from "./normalisation";

export type LayerSpecification =
	| InputLayerSpecification
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

export type Layer = InputLayer | LinearLayer | NonLinearLayer | NormalisationLayer | CostLayer;

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
	expectedOutputs: ActivationVector,
	inputs: ActivationVector,
	layer: Layer,
): Layer & ActivatedLayer => {
	switch (layer.kind) {
		case LayerKind.Input:
			return activateInputLayer(inputs, layer);
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
	subsequentLayers: readonly BackpropagatedLayer[],
): Layer & BackpropagatedLayer => {
	switch (layer.kind) {
		case LayerKind.Input:
			return backpropagateInputLayer(layer, subsequentLayers);
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
