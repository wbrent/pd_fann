# pd_fann

[fann] is a Pure Data interface to the FANN library for neural networks: http://leenissen.dk/fann/wp/. The starting point was David Morelli's [ann_mlp], which used FANN 1.0. Several updates and additions were implemented by William Brent in 2018 to conform with and take advantage of the FANN 2.0 API. A new Pd help file was also created specifically for [fann].

The FANN library has many functions for creating, configuring, training, running, and saving neural networks. [fann] accepts a variety of messages for executing FANN functions, making it possible to create and train a neural net incrementally in real time, or in batch mode using large datasets saved as separate files.
