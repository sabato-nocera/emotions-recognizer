<h1>
    Emotions Recognizer
</h1>

<h5>
    Progetto svolto nell'ambito del tirocinio curriculare <i>(Laurea Triennale in Informatica)</i>
</h5>


<p>

La directory del progetto è stata suddivisa nel seguente modo:

</p>

<ul>
	<li>
		<b>datasets</b>, al cui interno sono presenti i seguenti dataset utilizzati:
		<ul>
			<li>
				<i><b>full_dataset.csv</b></i>, il quale rappresenta il dataset originale, ricavato sulla base delle misure fisiologiche e comportamentali del campione che si è sottoposto all'esperimento.
			</li>
			<li>
				<i><b>full_dataset_without_humidity.csv</b></i>, il quale rappresenta il dataset originale senza la feature "Humidity", in quanto è stata evidenziata l'inaccuratezza del sensore utilizzato per il campionamento dell'umidità. Rispetto al <i> full_dataset.csv </i>, non sono stati riscontrati miglioramenti.			
			</li>
			<li>
				<i><b>full_dataset_without_humidity_augmented.csv</b></i>, il quale rappresenta una copia del dataset <i> full_dataset_without_humidity.csv </i> ampliato attraverso una tecnica di data augmentation per serie temporali. Il dataset originale è stato ottenuto campionando emozioni da ogni persona, ottenendo un totale di 80 esempi per ciascuna. La regola utilizzata per l'aumento dei dati è stato prendere coppie di esempi relativi allo stesso feedback di una stessa persona, suddividere le feature delle due tuple in due parti e creare due nuove tuple utilizzando parti alternate (similmente al meccanismo di crossing-over dei cromosomi). Usare tale dataset sembra portare a risultati leggermente migliori (sia in termini di accuracy che di loss); i valori di accuracy test ed accuracy train sono vicini, quindi l'aumento degli esempi porta ad una riduzione dell'overfitting.
			</li>
			<li>
				<i><b>full_dataset_without_humidity_augmented_balanced.csv</b></i>, il quale rappresenta una copia ridotta del dataset <i> full_dataset_without_humidity_augmented.csv </i>, in quanto ha un numero di esempi bilanciato per ogni classe. Usare tale dataset non sembra cambiare i risultati ottenuti, probabilmente perchè il numero di esempi scartati, rispetto al dataset su cui è basato, è piccolo.
			</li>
			<li>
				<i><b>full_dataset_without_humidity_reduced.csv</b></i>, il quale rappresenta risulta essere una copia ridotta del dataset <i> full_dataset_without_humidity.csv </i>, costituito dalle prime prime 47 * 80 osservazioni ( = 3760) di quest'ultimo. Usare tale dataset porta a risultati nettamente peggiori: drastica riduzione della test accuracy in tutti i modelli, aumento della loss nelle reti neurali che utilizzano la loss function <i> categorical crossentropy </i> e presenza di overfitting in tutte le reti neurali (ad eccezione delle CNN).
			</li>
		</ul>		
	</li>
	<li>
		<b> er_utils </b>, il quale costituisce un package con all'interno funzioni di utility.	
	</li>
	<li>
		<b> logs </b>, al cui interno sono presenti gli output prodotti dai sorgenti, in modo tale da poter essere sempre consultati e comparati con facilità.	
	</li>
	<li>
		<b> models </b>, al cui interno sono presenti i modelli risultanti dal training delle reti neurali.	
	</li>
	<li>
		<b> src </b>, al cui interno sono presenti le directory denominate con il nome del tipo di modello di Machine Learning di cui contengono varie configurazioni.
	</li>
</ul>

<h4>
    Osservazioni
</h4>

<p>
La loss function categorical categorical cross entropy risulta essere, per sua natura, quella più adatta al training di una rete neurale atta a riconoscere più classi. Nonostante ci siano state configurazioni con loss function MSE che hanno raggiunto ottimi risultati in termini di accuracy e loss, nella realtà il loro utilizzo potrebbe essere non del tutto conveniente, per questo si mantiene come riferimento la loss function categorical categorical cross entropy; infatti, le ROC curve  di stessi modelli, differenti solo per tali loss function, indicano che la categorical cross entropy ha una probabilità leggermente minore di individuare falsi positivi rispetto al MSE (quindi la prima risulta in realtà più sicura delle predizioni che compie).
</p>
<p>
Utilizzata la K-fold Cross Validation (convalida incrociata), tecnica statistica che consiste nella suddivisione dell' insieme di dati totale in k parti di uguale numerosità e, a ogni passo, la kª parte del'insieme di dati viene a essere quella di convalida, mentre la restante parte costituisce sempre l'insieme di addestramento. Così si allena il modello per ognuna delle k parti, evitando quindi problemi di sovradattamento, ma anche di campionamento asimmetrico (e quindi affetto da distorsione) del campione osservato, tipico della suddivisione dei dati in due sole parti (ossia addestramento/convalida). In altre parole, si suddivide il campione osservato in gruppi di egual numerosità, si esclude iterativamente un gruppo alla volta e si cerca di predirlo coi gruppi non esclusi, al fine di verificare la bontà del modello di predizione utilizzato. 
</p>
<p>
I Decision Tree (in questo caso, più appropriatamente Classification Tree), sono risultati essere il miglior modello utilizzato. Per il loro training è decisiva la scelta della massima profondità dell'albero; si è osservato che, man mano che si aumenta tale parametro, arriviamo ad un punto in cui non ha più senso aumentarlo ulteriormente, in quanto il modello non è più in grado di migliorare la propria accuracy. Rispetto alle reti neurali utilizzate, i Decision Tree sono risultati essere dei modelli che richiedono pochissimo tempo per il training e sicuri nelle predizioni che compiono (a prova di ciò, la curva di ROC è risulta assumere la forma di un angolo retto per ogni classe). 
</p> 

<h4>
    Punti d'interesse
</h4>

<ul>
    <li>
        Per ogni tipologia di rete neurale e per ogni dataset creato, è stato implementato un relativo modello. Lo scopo è confrontare in che modo stesse tipologie di reti neurali si comportano su diversi dataset e come diversi dataset vengono trattati da diverse tipologie di reti neurali. In questo modo, sarà anche possibile evidenziare l'importanza della quantità e della qualità degli esempi di un dataset ed approfondire il discorso circa la data augmentation.
    </li>
    <li>
        Confronto nell'utilizzo della K-Cross Validation con split pari a 5 e pari 10 tra i modelli implementati.
    </li>
    <li>
        Preso in considerazione il dataset originale senza la feature dell'umidità, ho implementato reti neurali MLP, LSTM, CNN e ResNet, tutte che utilizzano come loss function la Categorical Cross Entropy e come optimizer Adam, e le ho allenate utilizzando 4 diversi tipi di normalizzazione e senza normalizzazione. Lo scopo è studiare in che modo la normalizzazione "locale" dei dati campionati dai sensori risulta impattare sulle prestazioni delle reti neurali.
    </li>
    <li>
        Per la tipologia di rete neurale LSTM, sono stati implementati due modelli per ogni dataset, uno rappresentativo della LSTM bidirezionale e l'altro rappresentativo della LSTM non bidirezionale, al fine di confrontare le prestazioni delle due tipologie di LSTM su dati sensoristici come quelli che abbiamo a disposizione.
    </li>
    <li>
        Per la tipologia di rete neurale LSTM utilizzante il dataset senza umidità e la loss function Categorical Cross Entropy, sono stati implementati due modelli, uno rappresentativo della 1-layer LSTM ed uno della 3-layer LSTM, al fine di confrontare le prestazioni delle LSTM (non bidirezionali) con uno e tre layer su dati sensoristici come quelli che abbiamo a disposizione.
    </li>
    <li>
        Per la tipologia di rete neurale MLP, sono stati implementati due modelli per ogni dataset, uno che utilizza come loss function la Categorical Cross Entropy e l'altro che utilizza la Mean Squared Error, al fine di confrontare le prestazioni e l'affidabilità dei due modelli.
    </li>
</ul>