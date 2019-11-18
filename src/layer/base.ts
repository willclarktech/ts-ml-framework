export type Activation = number;
export type ActivationVector = readonly Activation[];

export type Delta = number;
export type DeltaVector = readonly Delta[];

export type Weight = number;
export type WeightVector = readonly Weight[];
export type WeightMatrix = readonly WeightVector[];

export type Bias = number;
export type BiasVector = readonly Bias[];

export enum LayerKind {
	Input = "input",
	Linear = "linear",
	NonLinear = "non-linear",
	Normalisation = "normalisation",
	Cost = "cost",
}

export interface BaseLayer {
	kind: LayerKind;
}

export interface ActivatedLayer extends BaseLayer {
	readonly inputs: ActivationVector;
	readonly activations: ActivationVector;
}

export interface BackpropagatedLayer extends BaseLayer {
	readonly deltas: DeltaVector;
}

export interface BaseLayerSpecification {
	readonly kind: LayerKind;
}
