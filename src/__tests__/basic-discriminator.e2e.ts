import { ActivationVector, LayerKind } from "../layer";
import { normalisationFunctionMap } from "../maths";
import { activateNetwork, Batch, createNetwork, OutputInputPair } from "../network";
import { getAverageError, train } from "../train";
import { unzip } from "../utils";

const logFrequency = 0;

test("4 categories dependent on 2 variables", () => {
	const [kangaroo, leopard, magpie, squirrel] = [
		[1, 0, 0, 0],
		[0, 1, 0, 0],
		[0, 0, 1, 0],
		[0, 0, 0, 1],
	];

	// Input: length (m), bipedal (boolean)
	const createKangaroo = (): ActivationVector => [(Math.random() * 60 + 100) / 100, 1];
	const createLeopard = (): ActivationVector => [(Math.random() * 106 + 90) / 100, 0];
	const createMagpie = (): ActivationVector => [(Math.random() * 2 + 20) / 100, 1];
	const createSquirrel = (): ActivationVector => [(Math.random() * 4 + 19) / 100, 0];

	const nTrainExamples = 100;
	const trainExamples = [...new Array(nTrainExamples)];
	const trainKangaroos = trainExamples.map(createKangaroo);
	const trainLeopards = trainExamples.map(createLeopard);
	const trainMagpies = trainExamples.map(createMagpie);
	const trainSquirrels = trainExamples.map(createSquirrel);
	const trainingSet: Batch = [
		...trainKangaroos.map((k: ActivationVector): OutputInputPair => [kangaroo, k]),
		...trainLeopards.map((l: ActivationVector): OutputInputPair => [leopard, l]),
		...trainMagpies.map((m: ActivationVector): OutputInputPair => [magpie, m]),
		...trainSquirrels.map((s: ActivationVector): OutputInputPair => [squirrel, s]),
	];
	const shuffledTrainingSet = trainingSet
		.map((c: OutputInputPair): readonly [number, OutputInputPair] => [Math.random(), c])
		.sort()
		.map(([_, c]) => c);
	const [trainOutputs, trainInputs] = unzip(shuffledTrainingSet);

	const nTestExamples = 5;
	const testExamples = [...new Array(nTestExamples)];
	const testKangaroos = testExamples.map(createKangaroo);
	const testLeopards = testExamples.map(createLeopard);
	const testMagpies = testExamples.map(createMagpie);
	const testSquirrels = testExamples.map(createSquirrel);
	const testingSet: Batch = [
		...testKangaroos.map((k: ActivationVector): OutputInputPair => [kangaroo, k]),
		...testLeopards.map((l: ActivationVector): OutputInputPair => [leopard, l]),
		...testMagpies.map((m: ActivationVector): OutputInputPair => [magpie, m]),
		...testSquirrels.map((s: ActivationVector): OutputInputPair => [squirrel, s]),
	];
	const [testOutputs, testInputs] = unzip(testingSet);

	const specifications = [
		{
			kind: LayerKind.Input as const,
			width: 2,
		},
		{
			kind: LayerKind.Linear as const,
			width: 5,
		},
		{
			kind: LayerKind.NonLinear as const,
			fn: "tanh" as const,
		},
		{
			kind: LayerKind.Linear as const,
			width: 4,
		},
		{
			kind: LayerKind.Normalisation as const,
			fn: "softmax" as const,
		},
		{
			kind: LayerKind.Cost as const,
			fn: "mean-squared-error" as const,
		},
	];
	const initialNetwork = createNetwork(specifications);
	const iterations = 200;
	const alpha = 0.001;
	const trained = train(initialNetwork, trainOutputs, trainInputs, iterations, alpha, logFrequency);
	const tested = activateNetwork(testOutputs, testInputs, trained);

	const error = getAverageError(tested);
	expect(error).toBeLessThan(0.01);

	const argmax = normalisationFunctionMap.get("argmax");
	if (!argmax) {
		throw new Error("argmax function not found");
	}
	const guesses = tested.layers[tested.layers.length - 2].activationsBatch.map(argmax.calculate);
	const correct = testOutputs.reduce((total, testOutput, i) => {
		const guess = guesses[i];
		return total + Number(testOutput.every((n, j) => n === guess[j]));
	}, 0);
	expect(correct).toEqual(testOutputs.length);
});
