<h1>
    Emotions Recognizer
</h1>

<h5>
    Progetto svolto nell'ambito del tirocinio curriculare <i>(Laurea Triennale in Informatica)</i>
</h5>

<hr/>

<p>
	L'obiettivo dell'attività svolta è stato confrontare l'accuracy raggiunta da diversi modelli di machine learning, atti al riconoscimento delle emozioni basato su sensor-data; nello specifico, ai fini della classificazione sono stati considerati parametri fisiologici (frequenza cardiaca, temperatura) e comportamentali (movimento del braccio, movimento della mano, tono della voce).
	
</p>	

Le emozioni prese in considerazione sono state quattro: 
	
<ul>
	<li>Felicità</li>
	<li>Tristezza</li>
	<li>Rabbia</li>
	<li>Paura</li>
</ul>

I modelli di machine learning utilizzati sono stati:

<ul>
	<li>Decision tree</li>
	<li>Multilayer perceptron (MLP)</li>
	<li>Residual neural network (ResNet)</li>
	<li>Convolutional neural network (CNN)</li>
	<li>Long short-term memory (LSTM)</li>
	<li>Long short-term memory bidirezionale (BiLSTM)</li>
	<li>Long short-term memory con tre layer (3-layer LSTM)</li>
	<li>Convolutional neural network + Long short-term memory (CNN LSTM)</li>
</ul>

<p>
	Per ciascuna tipologia di modello di machine learning sono state previste diverse implementazioni, ciascuna delle quali prevedeva una diversa strategia per la normalizzazione dei dati ed un diverso dataset per l'addestramento. Il modello di machine learning che ha raggiunto l'accuracy maggiore è stato il decision tree con un valore pari al <b>91.47%</b>.

<hr/>

La directory del progetto è stata suddivisa nel seguente modo:

</p>

<ul>
	<li>
		<b>datasets</b>, al cui interno sono presenti i seguenti dataset utilizzati;	
	</li>
	<li>
		<b>er_utils</b>, il quale costituisce un package con all'interno funzioni di utility;
	</li>
	<li>
		<b>logs</b>, al cui interno sono presenti gli output prodotti dell'esecuzione sorgenti, in modo tale da poter essere sempre consultati e comparati con facilità;	
	</li>
	<li>
		<b>models</b>, al cui interno sono presenti i modelli risultanti dal training delle reti neurali;	
	</li>
	<li>
		<b>src</b>, contenente i sorgenti implementati, in particolare le directory denominate con il nome di un modello di machine learning presentano al loro interno  varie configurazioni di quest'ultimo.
	</li>
</ul>
