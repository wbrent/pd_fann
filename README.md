# pd_fann

[fann] is a Pure Data interface to the FANN library for neural networks: http://leenissen.dk/fann/wp/ (GitHub repo: https://github.com/libfann/fann). The starting point was Davide Morelli's [ann_mlp], which is an object in the [ann] library by IOannes m zmolnig, Davide Morelli, and Georg Holzmann: https://github.com/sebshader/ann. Because [ann_mlp] was written for the FANN 1.0 API, it does not sucessfully compile against current FANN builds.

The [fann] object available here features updates and additions made by William Brent in 2018 to conform with and take advantage of the FANN 2.0 API. A detailed Pd help file was also created specifically for [fann]. Note that this repository does not contain updates of the other objects in the [ann] library (i.e., [ann_som] and [ann_td]). It is exclusively an interface to the FANN library meant to update and extend the original [ann_mlp] external.

The FANN library has many functions for creating, configuring, training, running, and saving neural networks. [fann] accepts a variety of messages for executing FANN functions, making it possible to create and train a neural net incrementally in real time, or in batch mode using large datasets saved as separate files.
