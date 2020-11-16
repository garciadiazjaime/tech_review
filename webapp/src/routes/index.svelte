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
	}

	function learning() {
		const game = output.length
		players = input[game]
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

	function areSamePlayers(data) {
		return players[0] === data[0] && players[1] === data[1] && players[2] === data[2]
	}

	async function anoterGame() {
		reset()
		const currentPlayers = [...players]
		while(areSamePlayers(currentPlayers)) {
			setRandomPlayers()
		}

		showLoader = true
		await setPredictedWinner()
		showLoader = false
	}

	async function setPredictedWinner() {
		console.log('players', players)
		const prediction = model.predict(tf.tensor([players]));
		prediction.print()
		
		learnings = await prediction.array()
	}

	function getWinner(data) {
		const response = data[0].reduce((accu, value, index) => {
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
		
		return ["Rock", "Paper", "Scissor"][response.index]
	}

	function getConfidence(data, value) {
		return `${Math.floor(data[0][value] * 100)} %`
	}

	async function traning() {
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
		if (gameOver) {
			return
		}

		const zeros = [0, 0, 0]
		zeros[indexSelected] = 1
		output = [...output, zeros]

		winner = indexSelected
		showLoader = true

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
		font-size: 42px;
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

	.btn {
		display: inline-block;
		padding: 10px 20px;
		border: 1px solid black;
		margin: 10px 0;
	}

	.btn:hover {
		cursor: pointer;
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
	<div class:selected="{players[0]}" class:winner="{0 === winner}" on:click={() => clickHandler(0)}>Rock</div>
	<div class:selected="{players[1]}" class:winner="{1 === winner}" on:click={() => clickHandler(1)}>Paper</div>
	<div class:selected="{players[2]}" class:winner="{2 === winner}" on:click={() => clickHandler(2)}>Scissor</div>
</section>

{#if showLoader }
<div class="loader"></div>
{/if}

{#if learnings} 
<div class="learnings">
	I've learned enought to say that <b>{getWinner(learnings)}</b> is the winner :)

	<br />
	Confidence Values:
	<table>
		<tr>
			<td>Rock</td>
			<td>{getConfidence(learnings, 0)}</td>
		</tr>
		<tr>
			<td>Paper</td>
			<td>{getConfidence(learnings, 1)}</td>
		</tr>
		<tr>
			<td>Scissor</td>
			<td>{getConfidence(learnings, 2)}</td>
		</tr>
	</table>

	<span class="btn" on:click={anoterGame}>Click me if you want me to predict another game</span>
</div>
{/if}
