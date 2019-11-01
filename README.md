# Rnn Character Prediction

A Keras character prediction machine learning project with the intention of using an rnn to generate Magic: The Gathering cards,
after training the model on a .csv produced from a JSON download on [scryfall](https://scryfall.com/docs/api/bulk-data). The training
file is [rnn.py](https://github.com/chimstead/rnn-character-prediction/blob/master/rnn.py), the text generation file is [rnn_generate.py](https://github.com/chimstead/rnn-character-prediction/blob/master/rnn_generate.py), and [cards_created.txt](https://github.com/chimstead/rnn-character-prediction/blob/master/cards_created.txt) contains some of the cream of the crop that the model has generated so far. While I built this with the intention of generating magic cards, these files can be appropriated to produce text in any style given enough (>100,000ish characters, ideally 1,000,000 or more) training data, so feel free to use these files to whatever text generation end you desire.
## Authors

* **Conor Himstead**

See also the [RoboRosewater](https://twitter.com/RoboRosewater) twitter account that inspired this mimic, and [The Unreasonable Effectiveness of Recurrant Neural Networks], which introduced a lot of folks to the power of character-level prediction. 
