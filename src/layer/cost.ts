import { CostFunction, costFunctionMap, CostFunctionName } from "../maths";
import { zipWith } from "../utils";
import {
	ActivatedLayer,
	ActivationVectorBatch,
	BackpropagatedLayer,
	BaseLayer,
	BaseLayerSpecification,
	LayerKind,
} from "./base";

export interface CostLayer extends BaseLayer {
	readonly kind: LayerKind.Cost;
	readonly fn: CostFunction;
	readonly expectedInputsBatch?: ActivationVectorBatch;
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
	expectedInputsBatch: ActivationVectorBatch,
	inputsBatch: ActivationVectorBatch,
	layer: CostLayer,
): CostLayer & ActivatedLayer => {
	return {
		...layer,
		expectedInputsBatch,
		inputsBatch,
		activationsBatch: zipWith(
			(expectedInputs, inputs) => [layer.fn.calculate(expectedInputs, inputs)],
			expectedInputsBatch,
			inputsBatch,
		),
	};
};

export const backpropagateCostLayer = (
	layer: CostLayer & ActivatedLayer,
	subsequentLayers: readonly BackpropagatedLayer[],
): CostLayer & BackpropagatedLayer => {
	if (subsequentLayers.length) {
		throw new Error("Cannot backpropagate cost layer with subsequent layers");
	}
	if (!layer.expectedInputsBatch) {
		throw new Error("Cannot backpropagate cost layer without expected inputs");
	}
	return {
		...layer,
		deltasBatch: zipWith(
			(expectedInputs, inputs) => layer.fn.derivative(expectedInputs, inputs),
			layer.expectedInputsBatch,
			layer.inputsBatch,
		),
	};
};

export const updateCostLayer = ({ kind, fn }: CostLayer & BackpropagatedLayer): CostLayer => ({
	kind,
	fn,
});
