export type Activation = number;
export type ActivationVector = readonly Activation[];
export type ActivationVectorBatch = readonly ActivationVector[];

export type Delta = number;
export type DeltaVector = readonly Delta[];
export type DeltaVectorBatch = readonly DeltaVector[];

export type Weight = number;
export type WeightVector = readonly Weight[];
export type WeightMatrix = readonly WeightVector[];

export type Bias = number;
export type BiasVector = readonly Bias[];

export enum LayerKind {
	Input = "input",
	Convolutional = "convolutional",
	Linear = "linear",
	NonLinear = "non-linear",
	Normalisation = "normalisation",
	Cost = "cost",
}

export interface BaseLayer {
	kind: LayerKind;
}

export interface ActivatedLayer extends BaseLayer {
	readonly inputsBatch: ActivationVectorBatch;
	readonly activationsBatch: ActivationVectorBatch;
}

export interface BackpropagatedLayer extends ActivatedLayer {
	readonly deltasBatch: DeltaVectorBatch;
}

export interface BaseLayerSpecification {
	readonly kind: LayerKind;
}

export type Alpha = number;
