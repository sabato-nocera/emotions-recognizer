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
				<i><b>full_dataset_without_humidity_reduced.csv</b></i>, il quale rappresenta risulta essere una copia ridotta del dataset <i> full_dataset_without_humidity.csv </i>, costituito dalle prime prime 47 * 80 osservazioni ( = 3760) di quest'ultimo.
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
