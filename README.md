<h1>
    Emotions Recognizer
</h1>

<h5>
    Project carried out as part of the curricular internship <i>(Bachelor's Degree in Computer Science)</i>
</h5>

<hr/>

<p>
	The goal of the project was to compare the accuracy achieved by different machine learning models designed for sensor-data-based emotion recognition; specifically, physiological (heart rate, temperature) and behavioral (arm movement, hand movement, tone of voice) parameters were considered for classification purposes.
	
</p>	

Four emotions were considered: 
	
<ul>
	<li>Happiness</li>
	<li>Sadness</li>
	<li>Anger</li>
	<li>Fear</li>
</ul>

The machine learning models employed were:

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
	Different configurations were implemented for each type of machine learning model, each with a different strategy for data normalization and a different dataset for training. The machine learning model that achieved the highest accuracy was the <b>decision tree</b> with a value of <b>91.47%</b>.

<hr/>

The project directory was divided as follows:

</p>

<ul>
	<li>
		<b>datasets</b>, containing the datasets used;
	</li>
	<li>
		<b>er_utils</b>, which constitutes a package with utility functions;
	</li>
	<li>
		<b>logs</b>, within which are the outputs produced by the source execution, so that they can always be consulted and compared with ease;	
	</li>
	<li>
		<b>models</b>, within which are the patterns resulting from the training of neural networks;	
	</li>
	<li>
		<b>src</b>, containing the source code of the implemented models -- in particular, directories named after a machine learning model have various configurations of that model within them.
	</li>
</ul>
