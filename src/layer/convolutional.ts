import { mean, transpose } from "../maths";
import { flatten, nest, zipWith } from "../utils";
import {
	ActivatedLayer,
	ActivationVector,
	ActivationVectorBatch,
	Alpha,
	BackpropagatedLayer,
	BaseLayer,
	BaseLayerSpecification,
	DeltaVector,
	DeltaVectorBatch,
	LayerKind,
	WeightMatrix,
} from "./base";
import {
	activateLinearLayer,
	createLinearLayer,
	LinearLayer,
	LinearLayerSpecification,
	updateLinearLayer,
} from "./linear";

export type Kernel = LinearLayer;
export type Kernels = readonly Kernel[];
export type ActivatedKernel = readonly (Kernel & ActivatedLayer)[];
export type ActivatedKernels = readonly ActivatedKernel[];
export type BackpropagatedKernel = readonly (Kernel & BackpropagatedLayer)[];

export interface ConvolutionalLayer extends BaseLayer {
	readonly kind: LayerKind.Convolutional;
	readonly width: number;
	readonly inputWidth: number;
	readonly kernels: Kernels;
}

export interface ActivatedConvolutionalLayer extends ActivatedLayer {
	readonly activatedKernels: ActivatedKernels;
}

export interface BackpropagatedConvolutionalLayer extends BackpropagatedLayer {
	readonly activatedKernels: ActivatedKernels;
}

export interface ConvolutionalLayerSpecification extends BaseLayerSpecification {
	readonly kind: LayerKind.Convolutional;
	readonly kernels: number;
	readonly convolutionSize: number;
	readonly inputWidth: number;
	readonly shouldPad?: boolean;
}

export const createConvolutionalLayer = (
	{ convolutionSize, inputWidth, kernels, shouldPad = true }: ConvolutionalLayerSpecification,
	previousWidth: number,
): ConvolutionalLayer => {
	if (convolutionSize !== 3) {
		throw new Error("Only convolution size 3 is supported");
	}
	if (!shouldPad) {
		throw new Error("Only convolutional layers with padding are supported");
	}
	const linearLayerSpecification: LinearLayerSpecification = {
		kind: LayerKind.Linear,
		width: 1,
	};
	return {
		kind: LayerKind.Convolutional,
		inputWidth,
		width: previousWidth,
		kernels: [...new Array(kernels)].map(
			createLinearLayer.bind(null, linearLayerSpecification, convolutionSize ** 2),
		),
	};
};

type ConvolutedActivation = readonly ActivationVector[];
type Rows = readonly (readonly number[])[];

export const take3FromRowAroundIndex = (i: number) => (row: readonly number[]): readonly number[] => {
	const start = i === 0 ? 0 : row[i - 1];
	const middle = row[i];
	const end = i === row.length - 1 ? 0 : row[i + 1];
	return [start, middle, end];
};

export const take3FromEachRelevantRow = (
	rows: Rows,
	rowIndex: number,
): ((_: unknown, columnIndex: number) => readonly number[]) => {
	const blankRow = new Array(rows[0].length).fill(0);
	return (_: unknown, columnIndex: number): readonly number[] => {
		const startRow = rowIndex === 0 ? blankRow : rows[rowIndex - 1];
		const middleRow = rows[rowIndex];
		const endRow = rowIndex === rows.length - 1 ? blankRow : rows[rowIndex + 1];
		const relevantRows = [startRow, middleRow, endRow];
		return flatten(relevantRows.map(take3FromRowAroundIndex(columnIndex)));
	};
};

export const getConvolutionsForRow = (rows: Rows) => (
	row: readonly number[],
	i: number,
): readonly (readonly number[])[] => row.map(take3FromEachRelevantRow(rows, i));

export const convoluteInputs = (inputWidth: number) => (inputs: ActivationVector): ConvolutedActivation => {
	const nRows = inputs.length / inputWidth;
	if (nRows % 1) {
		throw new Error("Input length is not a multiple of specified input width");
	}
	const rows = [...new Array(nRows)].map((_, i) => inputs.slice(i * inputWidth, (i + 1) * inputWidth));
	return flatten(rows.map(getConvolutionsForRow(rows)));
};

export const activateConvolutionalLayer = (
	inputsBatch: ActivationVectorBatch,
	layer: ConvolutionalLayer,
): ConvolutionalLayer & ActivatedConvolutionalLayer => {
	const convolutedInputsBatch: readonly ActivationVectorBatch[] = transpose(
		inputsBatch.map(convoluteInputs(layer.inputWidth)),
	);
	const activatedKernels: ActivatedKernels = layer.kernels.map(kernel =>
		convolutedInputsBatch.map(kernelInputsBatch => activateLinearLayer(kernelInputsBatch, kernel)),
	);

	const perKernelLayerActivationsBatch = activatedKernels.map(activatedKernel =>
		activatedKernel.reduce(
			(activationsBatch: ActivationVectorBatch, kernelActivationBatch) =>
				zipWith(
					(batch: ActivationVector, [activation]: ActivationVector) => {
						if (process.env.IMPURE) {
							(batch as number[]).push(activation);
							return batch;
						}
						return [...batch, activation];
					},
					activationsBatch,
					kernelActivationBatch.activationsBatch,
				),
			new Array(inputsBatch.length).fill([]),
		),
	);

	const layerActivationsBatch = transpose(perKernelLayerActivationsBatch).map(perBatchKernelActivations =>
		transpose(perBatchKernelActivations).map(mean),
	);

	return {
		...layer,
		inputsBatch,
		activatedKernels: activatedKernels,
		activationsBatch: layerActivationsBatch,
	};
};

const rotateWeights = (weights: WeightMatrix): WeightMatrix =>
	[...weights].reverse().map(weightVector => [...weightVector].reverse());

const rotateKernel = (kernel: LinearLayer): LinearLayer => ({
	...kernel,
	weights: rotateWeights(kernel.weights),
});

export const backpropagateConvolutionalLayer = (
	layer: ConvolutionalLayer & ActivatedConvolutionalLayer,
	[subsequentLayer]: readonly BackpropagatedLayer[],
): ConvolutionalLayer & BackpropagatedConvolutionalLayer => {
	if (!subsequentLayer) {
		throw new Error("Cannot backpropagate convolutional layer without subsequent layer");
	}

	const rotatedKernels = layer.kernels.map(rotateKernel);
	const convolutedDeltasBatch = transpose(subsequentLayer.deltasBatch.map(convoluteInputs(layer.inputWidth)));

	const activatedKernels: ActivatedKernels = rotatedKernels.map(rotatedKernel =>
		convolutedDeltasBatch.map(kernelDeltasBatch => activateLinearLayer(kernelDeltasBatch, rotatedKernel)),
	);
	const perKernelDeltasBatches = activatedKernels.map(activatedKernel =>
		activatedKernel.reduce(
			(deltasBatchAccumulator: DeltaVectorBatch, kernelActivationBatch) =>
				zipWith(
					(batch: DeltaVector, [activation]: DeltaVector) => {
						if (process.env.IMPURE) {
							(batch as number[]).push(activation);
							return batch;
						}
						return [...batch, activation];
					},
					deltasBatchAccumulator,
					kernelActivationBatch.activationsBatch,
				),
			new Array(subsequentLayer.deltasBatch.length).fill([]),
		),
	);
	const layerDeltasBatch = transpose(perKernelDeltasBatches).map(perBatchKernelDeltas =>
		transpose(perBatchKernelDeltas).map(kernelDeltas => mean(kernelDeltas) / kernelDeltas.length),
	);

	return {
		...layer,
		deltasBatch: layerDeltasBatch,
	};
};

export const updateConvolutionalLayer = (
	layer: ConvolutionalLayer & BackpropagatedConvolutionalLayer,
	deltasBatch: DeltaVectorBatch,
	alpha: Alpha,
): ConvolutionalLayer => {
	const combinedDeltasBatch = nest(flatten(deltasBatch));
	const perKernelInputsBatches = layer.activatedKernels.map(activatedKernel =>
		flatten(activatedKernel.map(k => k.inputsBatch)),
	);
	const updatedKernels = zipWith(
		(kernel, kernelInputsBatch) =>
			updateLinearLayer(
				{
					...kernel,
					activationsBatch: [],
					deltasBatch: [],
					inputsBatch: kernelInputsBatch,
				},
				combinedDeltasBatch,
				alpha,
			),
		layer.kernels,
		perKernelInputsBatches,
	);

	return {
		...layer,
		kernels: updatedKernels,
	};
};
