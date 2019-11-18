import {
	Matrix,
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
	ActivationVectorBatch,
	BackpropagatedLayer,
	BaseLayer,
	BaseLayerSpecification,
	Delta,
	DeltaVector,
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
	inputsBatch: ActivationVectorBatch,
	layer: NormalisationLayer,
): NormalisationLayer & ActivatedLayer => ({
	...layer,
	inputsBatch,
	activationsBatch: inputsBatch.map(inputs => layer.fn.calculate(inputs)),
});

export const backpropagateNormalisationLayer = (
	layer: NormalisationLayer & ActivatedLayer,
	[subsequentLayer]: readonly BackpropagatedLayer[],
): NormalisationLayer & BackpropagatedLayer => {
	if (!subsequentLayer) {
		throw new Error("Cannot backpropagate normalisation layer without subsequent layer");
	}
	const derivativesMatrices = layer.fn.derivativeInTermsOfOutput
		? layer.activationsBatch.map(layer.fn.derivativeInTermsOfOutput)
		: layer.inputsBatch.map(layer.fn.derivative);
	const weightedDerivativesMatrices = zipWith(
		(derivativesMatrix: Matrix, deltas: DeltaVector) =>
			zipWith(
				(delta: Delta, derivatives: Vector) => derivatives.map(derivative => derivative * delta),
				deltas,
				derivativesMatrix,
			),
		derivativesMatrices,
		subsequentLayer.deltasBatch,
	);
	return {
		...layer,
		deltasBatch: weightedDerivativesMatrices.map(weightedDerivativesMatrix =>
			transpose(weightedDerivativesMatrix).map(sum),
		),
	};
};

export const updateNormalisationLayer = ({
	kind,
	fn,
}: NormalisationLayer & BackpropagatedLayer): NormalisationLayer => ({
	kind,
	fn,
});
