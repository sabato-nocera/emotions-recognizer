# Emotions Recognizer


Progetto svolto nell'ambito del tirocinio curriculare (Laurea Triennale in Informatica)


La directory del progetto è stata suddivisa nel seguente modo:


• **datasets**, al cui interno sono presenti i quattro dataset utilizzati: 


_"full_dataset.csv"_ rappresenta il dataset originale, ricavato sulla base delle misure fisiologiche e comportamentali 
del campione che si è sottoposto all'esperimento.


_"full_dataset_without_humidity.csv"_ rappresenta il dataset originale senza la feature "Humidity", in quanto è stato
osservato che il sensore che campiona tale parametro potrebbe essere inaccurato; si è deciso di addestrare reti neurali
utilizzando anche tale dataset con lo speranza di osservare un miglioramento dei risultati. Fino ad ora, sono stati
riscontrati risultati analoghi ai precedenti.


_"reduced_dataset.csv"_ risulta essere una riduzione del dataset originale, privato di un certo numero di esempi, in 
modo tale da disporre dello stesso numero di esempi per feedback. La motivazione di tale scelta è che, in accordo a vari
studi, la predizione delle classi risulterebbe essere più accurata. Rispetto al dataset originale, la dimensione risulta
essere diminuita non di molto e, nel complesso, gli esempi risultano essere ancora esigui. Probabilmente, è anche per 
questo che, fino ad ora, sono stati riscontrati risultati analoghi ai precedenti.


_"sorted_dataset.csv"_ rappresenta il dataset originale, ordinato in accordo al feedback. Tale dataset è stato 
utilizzato in un processo di data augmentation, il quale ha previsto l'aggiunta di una nuova tupla ogni _k_ tuple 
rappresentative dello stesso feedback; tale tupla è risultata essere la media delle features delle tuple prese in esame. 
Fino ad ora, sono stati riscontrati risultati analoghi ai precedenti; probabilmente, ciò è dovuto sia alla dimensione
del dataset, la quale non è aumentata di molto, sia della "regola" utilizzata per la data augmentation, effettivamente
poco efficace.


_"second_augmented_dataset.csv"_ rappresenta il dataset originale senza la feature "Humidity", ampliato attraverso una
tecnica di data augmentation (per serie temporali, diversamente dalla prima strategia di data augmentation utilizzata). 
Il dataset originale è stato ottenuto campionando emozioni da ogni persona, ottenendo un totale di 80 esempi per 
ciascuna. La regola utilizzata per l'aumento dei dati è stato prendere coppie di esempi relativi allo stesso feedback di 
una stessa persona, suddividere le feature delle due tuple in due parti e creare due nuove tuple utilizzando parti 
alternate (similmente al meccanismo di crossing-over dei cromosomi). Usare tale dataset sembra portare a risultati 
leggermente migliori (sia in termini di accuracy che di loss); i valori di accuracy test ed accuracy train sono vicini.
La _MLP_ che lo utilizza raggiunge il 90.99% di accuracy test, con un valore di loss basso; ciò è dovuto anche grazie
all'aumento delle epoche e della batch size (sembra essere una buona strategia aumentare queste ultime due nelle reti
neurali che utilizzano tale dataset).


_"third_augmented_dataset.csv"_ rappresenta una copia di _"second_augmented_dataset.csv"_, ma con un numero di esempi
bilanciato per ogni classe.


• **logs**, al cui interno sono presenti gli output prodotti a seguito del training delle reti neurali, in modo tale da
poter essere sempre consultati e comparati con facilità.


• **models**, al cui interno sono presenti i modelli risultanti dal training delle reti neurali.


• **src**, al cui interno sono presenti le directory denominate con il nome del tipo di rete neurale di cui contengono i
vari codici sorgenti (ognuno rappresentativo di una diversa configurazione del tipo di rete neurale).


## Osseravazioni


Le reti neurali che vengono configurate con la loss function _categorical hinge_ impiegano poco tempo tempo per il 
training; benché complessivamente l'accuracy raggiunta è sempre stata molto bassa, è risultato essere controproducente 
l'inserimento di molti neuroni e layer intermedi.


La loss function _categorical categorical cross entropy_ risulta essere, per sua natura, quella più adatta al training
di una rete neurale atta a riconoscere più classi. Nonostante ci siano state configurazioni con loss function _MSE_ che
hanno performato meglio, nella realtà il loro utilizzo potrebbe essere non del tutto conveniente, per questo si mantiene
come riferimento la loss function _categorical categorical cross entropy_. Una prova di ciò si evince anche dall'
osservazione della ROC curve nelle reti _MLP_ che utilizzano  _"second_augmented_dataset.csv"_: avendo aumentato il
numero di esempi, notiamo come i valori di accuracy test ed accuracy train sono simili sia nella rete neurale che
utilizza come loss function _categorical categorical cross entropy_ che in quella che usa _MSE_; come possiamo ben
aspettarci, _MSE_ continua ad avere il valore di loss minore, nonostante sia simile a quello di stesse reti neurali che 
utilizzano diversi dataset, ma quello del _categorical categorical cross entropy_ risulta essere minore rispetto a di 
stesse reti neurali che utilizzano diversi dataset; inoltre, le ROC curve indicano che il _categorical categorical cross 
entropy_ ha una probabilità leggermente minore di individuare falsi positivi rispetto al _MSE_ (quindi nella realtà è 
leggermente più affidabile).


Utilizzata la _K-fold Cross Validation_ (convalida incrociata), tecnica statistica che consiste nella suddivisione dell'
insieme di dati totale in k parti di uguale numerosità e, a ogni passo, la kª parte del'insieme di dati viene a essere 
quella di convalida, mentre la restante parte costituisce sempre l'insieme di addestramento. Così si allena il modello 
per ognuna delle k parti, evitando quindi problemi di sovradattamento, ma anche di campionamento asimmetrico (e quindi 
affetto da distorsione) del campione osservato, tipico della suddivisione dei dati in due sole parti (ossia 
addestramento/convalida). In altre parole, si suddivide il campione osservato in gruppi di egual numerosità, si esclude 
iterativamente un gruppo alla volta e si cerca di predirlo coi gruppi non esclusi, al fine di verificare la bontà del 
modello di predizione utilizzato. Sarà curioso osservare i risultati che produrrà la _K-fold Cross Validation_ per stesse
reti neurali che utilizzano, però, dataset diversi. Purtroppo, tale tecnica necessità di eseguire il training molte volte,
portando ad un utilizzo elevato dell'elaboratore; maggiore è il numero dei split e maggiore sarà anche la stima di tale
tecnica, ma il costo computazionale aumenta considerevolmente. Per ogni reti neurale, è possibile richiamare un'apposita
funzione che effettua _K-fold Cross Validation_ per tale rete neurale.