export const printMnistExample = (input: readonly number[]): string => {
	const mnistWidth = 28;
	return [...new Array(mnistWidth)]
		.map((_, i) =>
			input
				.slice(i * mnistWidth, (i + 1) * mnistWidth)
				.map(n => (n > 127 ? "#" : "."))
				.join(""),
		)
		.join("\n");
};
