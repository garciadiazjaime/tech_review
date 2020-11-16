<script>
	import { onMount } from 'svelte';
	import * as tf from '@tensorflow/tfjs';

	let model = null
	let players = []
	let winner = null
	let showLoader = false
	let learnings = null
	let winnerPredicted = null
	let gameOver = false

	const input = [
		[1, 1, 0],
		[1, 0, 1],
		[0, 1, 1]
	]

	let output = []
	let tmp = []

	function getPlayer() {
		return Math.floor(Math.random() * 3); 
	}

	function setRandomPlayers() {
		const player1 = getPlayer()
		let player2 = getPlayer()
		while (player2 === player1) {
			player2 = getPlayer()
		}

		const response = [0, 0, 0]
		response[player1] = 1
		response[player2] = 1

		players = response
		console.log('players', players)
	}

	function learning() {
		const game = output.length
		players = input[game]
		console.log(players)
	}

	function reset() {
		winner = null
		showLoader = false
	}

	async function main() {
		if (output.length < 3) {
			reset()
			return learning()
		}

		gameOver = true

		await traning()
		reset()
		setRandomPlayers()
		await setPredictedWinner()

	}

	async function setPredictedWinner() {
		const prediction = model.predict(tf.tensor([players]));
		prediction.print()
		
		learnings = await prediction.array()
	}

	function getWinner() {
		const response = learnings[0].reduce((accu, value, index) => {
			if (value > accu.value) {
				accu = {
					value,
					index
				}
			}

			return accu
		}, {
			value: Number.NEGATIVE_INFINITY,
			index: null
		})
		console.log('response', response)
		
		return ["Rock", "Paper", "Scissor"][response.index]
	}

	function getConfidence(value) {
		return `${Math.floor(learnings[0][value] * 100)} %`
	}

	async function traning() {
		console.log('traning')
		model = tf.sequential();
		model.add(tf.layers.dense({units: 3, inputShape: [3]}));

		model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

		const xs = tf.tensor(input);
		const ys = tf.tensor(output);

		await model.fit(xs, ys, {epochs: 500})
		console.log('traning done')
	}

	onMount(async () => {
		main()
	});

	function clickHandler(indexSelected) {
		console.log('clickHandler')
		if (gameOver) {
			return
		}

		const zeros = [0, 0, 0]
		zeros[indexSelected] = 1
		output = [...output, zeros]
		console.log(output.length)

		winner = indexSelected
		showLoader = true

		console.log(JSON.stringify(input))
		console.log(JSON.stringify(output))

		setTimeout(() => {
			main()
		}, 500);
	}

</script>

<style>
	section {
		display: flex;
		justify-content: space-between;
	}

	section div {
		border: 1px dotted black;
		max-width: 30%;
		flex: 1;
		height: 200px;
		display: flex;
		align-items: center;
		justify-content: center;
		visibility: hidden;
	}

	section div.selected:hover {
		cursor: pointer;
	}

	.selected {
		border: 5px solid black;
		visibility: visible;
	}

	.winner {
		background-color: #c5efc5;
	}

	.off {
		background: none;
		border: 5px solid gainsboro;
		visibility: visible;
	}

	.learnings {
		margin: 20px 0;
	}

	.loader {
		border: 16px solid #f3f3f3; /* Light grey */
		border-top: 16px solid #3498db; /* Blue */
		border-radius: 50%;
		width: 60px;
		height: 60px;
		animation: spin 2s linear infinite;
		margin: 20px 0;
	}

	@keyframes spin {
		0% { transform: rotate(0deg); }
		100% { transform: rotate(360deg); }
	}
</style>

<svelte:head>
	<title>Sapper project template</title>
</svelte:head>

<h1>Pick the winner!</h1>

<section>
	<div class:selected="{players[0] && !gameOver}" class:winner="{0 === winner && !gameOver}" class:off="{gameOver}" on:click={() => clickHandler(0)}>Rock</div>
	<div class:selected="{players[1] && !gameOver}" class:winner="{1 === winner && !gameOver}" class:off="{gameOver}" on:click={() => clickHandler(1)}>Paper</div>
	<div class:selected="{players[2] && !gameOver}" class:winner="{2 === winner && !gameOver}" class:off="{gameOver}" on:click={() => clickHandler(2)}>Scissor</div>
</section>

{#if showLoader }
<div class="loader"></div>
{/if}

{#if learnings} 
<div class="learnings">
	I've learned enought to say that <b>{getWinner()}</b> is the winner :)

	<br />
	Confidence Values:
	<table>
		<tr>
			<td>Rock</td>
			<td>{getConfidence(0)}</td>
		</tr>
		<tr>
			<td>Paper</td>
			<td>{getConfidence(1)}</td>
		</tr>
		<tr>
			<td>Scissor</td>
			<td>{getConfidence(2)}</td>
		</tr>
	</table>
</div>
{/if}
