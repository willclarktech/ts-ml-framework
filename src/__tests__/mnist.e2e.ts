import { readFileSync } from "fs";

import { LayerKind } from "../layer";
import { normalisationFunctionMap } from "../maths";
import { activateNetwork, Batch, createNetwork, OutputInputPair } from "../network";
import { train, trainWithMiniBatches } from "../train";
import { createOneHot, unzip } from "../utils";

const logFrequency = 0;

const argmax = normalisationFunctionMap.get("argmax");
if (!argmax) {
	throw new Error("argmax function not found");
}

const linearNormalisation = normalisationFunctionMap.get("linear");
if (!linearNormalisation) {
	throw new Error("linear normalisation function not found");
}

const loadMnistData = (path: string): Batch => {
	const rawData = readFileSync(path, { encoding: "utf8" });
	// Line 0 is header; last line is blank
	const lines = rawData.split("\n").slice(1, -1);
	return lines.map(
		(line: string): OutputInputPair => {
			const [nextExpectedOutputIndex, ...nextInput] = line
				.split(",")
				.map((n: string): number => parseInt(n, 10));
			const nextExpectedOutput = createOneHot(nextExpectedOutputIndex);
			const nextInputNormalised = linearNormalisation.calculate(nextInput);
			return [nextExpectedOutput, nextInputNormalised];
		},
	);
};

// const trainDataPath = "./data/mnist/train.csv";
const trainDataPath = "./data/mnist/train.partial.csv";
const mnistData = loadMnistData(trainDataPath);
const nExamples = 1000;

const nTrainData = nExamples * 0.9;
const trainData = mnistData.slice(0, nTrainData);
const [trainOutputs, trainInputs] = unzip(trainData);
// const [testOutputs, testInputs] = [trainOutputs, trainInputs];
const testData = mnistData.slice(nTrainData);
const [testOutputs, testInputs] = unzip(testData);

test("mnist", () => {
	const specifications = [
		{
			kind: LayerKind.Input as const,
			width: 784,
		},
		{
			kind: LayerKind.Linear as const,
			width: 10,
		},
		{
			kind: LayerKind.Normalisation as const,
			fn: "softmax" as const,
		},
		{
			kind: LayerKind.Cost as const,
			fn: "cross-entropy" as const,
		},
	];
	const initialNetwork = createNetwork(specifications);
	const iterations = 70;
	const alpha = 0.002;
	const trained = train(initialNetwork, trainOutputs, trainInputs, iterations, alpha, logFrequency);
	const trainedFinal = activateNetwork(trainOutputs, trainInputs, trained);

	const trainedGuesses = trainedFinal.layers[trained.layers.length - 2].activationsBatch.map(
		argmax.calculate,
	);
	const trainedCorrect = trainOutputs.reduce((total, trainOutput, i) => {
		const guess = trainedGuesses[i];
		return total + Number(trainOutput.every((n, j) => n === guess[j]));
	}, 0);
	expect(trainedCorrect / trainOutputs.length).toBeGreaterThan(0.9);

	const tested = activateNetwork(testOutputs, testInputs, trained);

	const guesses = tested.layers[tested.layers.length - 2].activationsBatch.map(argmax.calculate);
	const correct = testOutputs.reduce((total, testOutput, i) => {
		const guess = guesses[i];
		return total + Number(testOutput.every((n, j) => n === guess[j]));
	}, 0);
	expect(correct / testOutputs.length).toBeGreaterThan(0.8);
});

test("mnist with mini-batches", () => {
	const specifications = [
		{
			kind: LayerKind.Input as const,
			width: 784,
		},
		{
			kind: LayerKind.Linear as const,
			width: 10,
		},
		{
			kind: LayerKind.Normalisation as const,
			fn: "softmax" as const,
		},
		{
			kind: LayerKind.Cost as const,
			fn: "cross-entropy" as const,
		},
	];
	const initialNetwork = createNetwork(specifications);
	const iterations = 40;
	const alpha = 0.01;
	const miniBatchSize = 32;
	const trained = trainWithMiniBatches(
		initialNetwork,
		trainOutputs,
		trainInputs,
		iterations,
		alpha,
		logFrequency,
		miniBatchSize,
	);
	const trainedFinal = activateNetwork(trainOutputs, trainInputs, trained);

	const trainedGuesses = trainedFinal.layers[trained.layers.length - 2].activationsBatch.map(
		argmax.calculate,
	);
	const trainedCorrect = trainOutputs.reduce((total, trainOutput, i) => {
		const guess = trainedGuesses[i];
		return total + Number(trainOutput.every((n, j) => n === guess[j]));
	}, 0);
	expect(trainedCorrect / trainOutputs.length).toBeGreaterThan(0.9);

	const tested = activateNetwork(testOutputs, testInputs, trained);

	const guesses = tested.layers[tested.layers.length - 2].activationsBatch.map(argmax.calculate);
	const correct = testOutputs.reduce((total, testOutput, i) => {
		const guess = guesses[i];
		return total + Number(testOutput.every((n, j) => n === guess[j]));
	}, 0);
	expect(correct / testOutputs.length).toBeGreaterThan(0.8);
});

test("mnist with mini-batches and hidden layer", () => {
	const specifications = [
		{
			kind: LayerKind.Input as const,
			width: 784,
		},
		{
			kind: LayerKind.Linear as const,
			width: 30,
		},
		{
			kind: LayerKind.NonLinear as const,
			fn: "tanh" as const,
		},
		{
			kind: LayerKind.Linear as const,
			width: 10,
		},
		{
			kind: LayerKind.Normalisation as const,
			fn: "softmax" as const,
		},
		{
			kind: LayerKind.Cost as const,
			fn: "cross-entropy" as const,
		},
	];
	const initialNetwork = createNetwork(specifications);
	const iterations = 40;
	const alpha = 0.01;
	const miniBatchSize = 32;
	const trained = trainWithMiniBatches(
		initialNetwork,
		trainOutputs,
		trainInputs,
		iterations,
		alpha,
		logFrequency,
		miniBatchSize,
	);
	const trainedFinal = activateNetwork(trainOutputs, trainInputs, trained);

	const trainedGuesses = trainedFinal.layers[trained.layers.length - 2].activationsBatch.map(
		argmax.calculate,
	);
	const trainedCorrect = trainOutputs.reduce((total, trainOutput, i) => {
		const guess = trainedGuesses[i];
		return total + Number(trainOutput.every((n, j) => n === guess[j]));
	}, 0);
	expect(trainedCorrect / trainOutputs.length).toBeGreaterThan(0.9);

	const tested = activateNetwork(testOutputs, testInputs, trained);

	const guesses = tested.layers[tested.layers.length - 2].activationsBatch.map(argmax.calculate);
	const correct = testOutputs.reduce((total, testOutput, i) => {
		const guess = guesses[i];
		return total + Number(testOutput.every((n, j) => n === guess[j]));
	}, 0);
	expect(correct / testOutputs.length).toBeGreaterThan(0.8);
});
