import { CostFunction, costFunctionMap, CostFunctionName } from "../maths";
import {
	ActivatedLayer,
	ActivationVector,
	BackpropagatedLayer,
	BaseLayer,
	BaseLayerSpecification,
	LayerKind,
} from "./base";

export interface CostLayer extends BaseLayer {
	readonly kind: LayerKind.Cost;
	readonly fn: CostFunction;
	readonly expectedInputs?: ActivationVector;
}

export interface CostLayerSpecification extends BaseLayerSpecification {
	readonly kind: LayerKind.Cost;
	readonly fn: CostFunctionName;
}

export const createCostLayer = ({ fn }: CostLayerSpecification): CostLayer => {
	const costFn = costFunctionMap.get(fn);
	if (!costFn) {
		throw new Error("Cannot create cost layer");
	}
	return {
		kind: LayerKind.Cost,
		fn: costFn,
	};
};

export const activateCostLayer = (
	expectedInputs: ActivationVector,
	inputs: ActivationVector,
	layer: CostLayer,
): CostLayer & ActivatedLayer => {
	if (expectedInputs.length !== inputs.length) {
		throw new Error("Cannot activate cost layer with mismatching expected inputs and inputs");
	}
	return {
		...layer,
		expectedInputs,
		inputs,
		activations: [layer.fn.calculate(expectedInputs, inputs)],
	};
};

export const backpropagateCostLayer = (
	layer: CostLayer & ActivatedLayer,
	subsequentLayers: readonly BackpropagatedLayer[],
): CostLayer & BackpropagatedLayer => {
	if (subsequentLayers.length) {
		throw new Error("Cannot backpropagate cost layer with subsequent layers");
	}
	if (!layer.expectedInputs) {
		throw new Error("Cannot backpropagate cost layer without expected inputs");
	}
	return {
		...layer,
		deltas: layer.fn.derivative(layer.expectedInputs, layer.inputs),
	};
};

export const updateCostLayer = ({ kind, fn }: CostLayer & BackpropagatedLayer): CostLayer => ({
	kind,
	fn,
});
