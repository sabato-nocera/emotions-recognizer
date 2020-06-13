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
leggermente migliori (sia in termini di accuracy che di loss).


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
come riferimento la loss function _categorical categorical cross entropy_.
