# pd_fann

[fann] is a Pure Data interface to the FANN library for neural networks: http://leenissen.dk/fann/wp/ (GitHub repo: https://github.com/libfann/fann). The starting point was Davide Morelli's [ann_mlp], which is an object in the [ann] library by IOannes m zmolnig, Davide Morelli, and Georg Holzmann: https://github.com/sebshader/ann. Because [ann_mlp] was written for the FANN 1.0 API, it does not sucessfully compile against current FANN builds.

The [fann] object available here features updates and additions made by William Brent in 2018 to conform with and take advantage of the FANN 2.0 API. A detailed Pd help file was also created specifically for [fann]. Note that this repository does not contain updates of the other objects in the [ann] library (i.e., [ann_som] and [ann_td]). It is exclusively an interface to the FANN library meant to update and extend the original [ann_mlp] external.

The FANN library has many functions for creating, configuring, training, running, and saving neural networks. [fann] accepts a variety of messages for executing FANN functions, making it possible to create and train a neural net incrementally in real time, or in batch mode using large datasets saved as separate files.

#### Compiling from source

The [fann] Pd external should be statically linked to the FANN library, which must be compiled from source. After cloning the FANN repository linked above, navigate to the fann directory. Under macOS, configure CMake with:

`cmake . -DCMAKE_OSX_SYSROOT=$(xcrun --sdk macosx --show-sdk-path)`

This tells CMake to use the SDK provided by Xcode so that all necessary headers are found. Under Linux, make sure to configure the build with position independent code (PIC):

`cmake . -DCMAKE_POSITION_INDEPENDENT_CODE=ON`

Next, run:

`make fann_static`

This ensures that libfann.a is produced in the build. Next, install the FANN library with:

`sudo make install`

Once libfann.a is compiled and installed, navigate to the pd\_fann directory and run:

`make`

At this point, you should be able to open the fann-help.pd patch and begin using the object.
