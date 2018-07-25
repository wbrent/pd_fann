/*	
	ann_mlp : Neural Networks for PD
	by Davide Morelli - info@davidemorelli.it - http://www.davidemorelli.it
	this software is simply an interface for FANN classes
	http://fann.sourceforge.net/
	FANN is obviously needed for compilation
	this software is licensed under the GNU General Public License
*/

/*
	hacked by Georg Holzmann for some additional methods, bug fixes, ...
	2005, grh@mur.at
*/

/*
	updated for FANN 2.0.0 by William Brent, 2018, w@williambrent.com
	
	TODO:
	- change unnecessary A_GIMME's to A_DEFFLOAT or A_DEFSYMBOL
	- method for fann_test()
	- route errors to Pd post window
	- method for writing errors to a file via fann_set_error_log(NULL, filePtr)
*/

#include <stdio.h>
#include <string.h>
#include "m_pd.h"
#include "fann.h"

#ifndef VERSION 
#define VERSION "0.3"
#endif

#ifndef __DATE__ 
#define __DATE__ ""
#endif

// WB: should really check for valid char pointers in FANN_ALGO_NAMES and FANN_ACTIVATIONFUNC_NAMES in order to determine how many there are, but hard coding these now based on documentation for FANN 2.0.0
#define NUMTRAININGALGOS 5
#define NUMACTIVATIONFUNCS 18
#define NUMTRAINERRORFUNCS 2
#define NUMTRAINSTOPFUNCS 2

static t_class *fann_class;

typedef enum
{
	false = 0,
	true
} t_bool;

typedef enum
{
	TRAIN = 0,
	RUN
} t_mode;

typedef struct _fann {
	t_object  x_obj;
    t_symbol *x_objSymbol;
	struct fann *x_ann;
	t_mode x_mode;
	t_symbol *x_fileName; // name of the file where this ann is saved
	t_symbol *x_fileNameTrain; // name of the file with training data
	t_float x_desiredError;
	unsigned int x_maxIterations;
	unsigned int x_iterationsBetweenReports;
	t_bool x_postCallback;
	fann_type *x_input;     // grh: storage for input
	fann_type *x_outputFann; // grh: storage for output (fann_type)
	t_atom *x_outputAtom;       // grh: storage for output (t_atom)
	t_canvas *x_canvas;
	t_outlet *x_listOut;
	t_outlet *x_epochsOut;
	t_outlet *x_mseOut;
} t_fann;


// callback function for mid-training info. this is called in fann_train_data.c, line 285, within fann_train_on_data()
int FANN_API ann_custom_callback(struct fann *ann, struct fann_train_data *train, unsigned int max_epochs, unsigned int epochs_between_reports, t_float desired_error, unsigned int epochs)
{
	t_fann *x;
	t_float currentMSE;
	
	x = (t_fann *)(ann->user_data);
	
	currentMSE = fann_get_MSE(ann);
	
	if(x->x_postCallback)
		post("fann: epochs: %i\t\t MSE: %f", epochs, currentMSE);
	
	outlet_float(x->x_mseOut, currentMSE);
	outlet_float(x->x_epochsOut, epochs);
	
	return 0;
}


// allocation
static void pdFann_allocateStorage(t_fann *x)
{
	unsigned int i, numInput, numOutput;

	if(!x->x_ann)
		return;

	numInput = fann_get_num_input(x->x_ann);
	numOutput = fann_get_num_output(x->x_ann);
	
	x->x_input = (fann_type *)getbytes(numInput*sizeof(fann_type));
	x->x_outputFann = (fann_type *)getbytes(numOutput*sizeof(fann_type));
	x->x_outputAtom = (t_atom *)getbytes(numOutput*sizeof(t_atom));

	// init storage with zeros
	for (i=0; i<numInput; i++)
		x->x_input[i] = 0;
		
	for (i=0; i<numOutput; i++)
	{
		x->x_outputFann[i] = 0;
		SETFLOAT(x->x_outputAtom+i, 0);
	}
}


// deallocation
static void pdFann_free(t_fann *x)
{
	unsigned int numInput, numOutput;
	
	if(!x->x_ann)
		return;

	numInput = fann_get_num_input(x->x_ann);
	numOutput = fann_get_num_output(x->x_ann);
	
	// free dynamic memory
	t_freebytes(x->x_input, numInput * sizeof(fann_type));
	t_freebytes(x->x_outputFann, numOutput * sizeof(fann_type));
	t_freebytes(x->x_outputAtom, numOutput * sizeof(t_atom));
	
	// destroy the ANN
	fann_destroy(x->x_ann);
}


static void pdFann_print(t_fann *x)
{
	if(!x->x_ann)
		pd_error(x, "%s: ANN is not initialized", x->x_objSymbol->s_name);
	else
	{
		unsigned int *neuronsPerLayer, numLayers, i;

		numLayers = fann_get_num_layers(x->x_ann);

		neuronsPerLayer = (unsigned int *)getbytes(numLayers*sizeof(unsigned int));

		// init to zero before getting neurons per layer
		for(i=0; i<numLayers; i++)
			neuronsPerLayer[i] = 0;

		fann_get_layer_array(x->x_ann, neuronsPerLayer);
		
		post("%s: ANN properties:", x->x_objSymbol->s_name);
		post("network type\t\t\t\t%s", FANN_NETTYPE_NAMES[fann_get_network_type(x->x_ann)]);
		
		if(x->x_mode)
			post("status\t\t\t\t%s", "RUNNING");
		else
			post("status\t\t\t\t%s", "TRAINING");

		post("total layers (including input/output layers)\t\t\t\t%i", fann_get_num_layers(x->x_ann));
			
		post("input layer [0] size\t\t\t\t%i", fann_get_num_input(x->x_ann));

		for(i=1; i<numLayers-1; i++)
			post("hidden layer [%i] size\t\t\t\t%i", i, neuronsPerLayer[i]);

		post("output layer [%i] size\t\t\t\t%i", numLayers-1, fann_get_num_output(x->x_ann));
		post("connection rate\t\t\t\t%f", fann_get_connection_rate(x->x_ann));
		post("learning rate\t\t\t\t%f", fann_get_learning_rate(x->x_ann));
		post("total neurons\t\t\t\t%i", fann_get_total_neurons(x->x_ann));
		post("total connections\t\t\t\t%i", fann_get_total_connections(x->x_ann));
		post("training algorithm\t\t\t\t%s", FANN_TRAIN_NAMES[fann_get_training_algorithm(x->x_ann)]);
		
		post("iterations between reports\t\t\t\t%i", x->x_iterationsBetweenReports);
		post("max iterations\t\t\t\t%i", x->x_maxIterations);
		
		post("activation function hidden\t\t\t\t%s", FANN_ACTIVATIONFUNC_NAMES[fann_get_activation_function(x->x_ann, 1, 0)]);
		post("activation function output\t\t\t\t%s", FANN_ACTIVATIONFUNC_NAMES[fann_get_activation_function(x->x_ann, numLayers-1, 0)]);	

		post("activation steepness hidden\t\t\t\t%f", fann_get_activation_steepness(x->x_ann, 1, 0));
		post("activation steepness output\t\t\t\t%f", fann_get_activation_steepness(x->x_ann, numLayers-1, 0));

		post("training error function\t\t\t\t%s", FANN_ERRORFUNC_NAMES[fann_get_train_error_function(x->x_ann)]);
		post("training stop function\t\t\t\t%s", FANN_STOPFUNC_NAMES[fann_get_train_stop_function(x->x_ann)]);

		if(x->x_postCallback)
			post("post callback\t\t\t\t%s", "TRUE");
		else
			post("post callback\t\t\t\t%s", "FALSE");

		// free the neuronsPerLayer memory
		t_freebytes(neuronsPerLayer, numLayers*sizeof(unsigned int));
		
// this doesn't work. Not sure why. Seems that x_ann->errstr is never assigned a new value
//		post("%s: last error\t\t\t\t%s", x->x_objSymbol->s_name, x->x_ann->errstr);
	}
}


static void pdFann_printParameters(t_fann *x)
{
	if(!x->x_ann)
		pd_error(x, "%s: ANN is not initialized", x->x_objSymbol->s_name);
	else
	{
		fann_print_parameters(x->x_ann);
	}
}


static void pdFann_createFann(t_fann *x, t_symbol *s, int argc, t_atom *argv)
{
	unsigned int numInput, numOutput, numLayers, *neuronsPerLayer;
	t_bool activated;
	int i, countArgs;
	t_float connectionRate, learningRate;
  
  	numInput = 2;
	numOutput = 1;
	numLayers = 3;
  	activated = false;
  	i = 0;
  	countArgs = 0;
  	connectionRate = 1.0;
  	learningRate = 0.7;
  	
	// okay, start parsing init args ...
	// argument 1: units in input layer
	if(argc > countArgs++)
		numInput = atom_getint(argv++);

	// argument 2: units in output layer
	if(argc > countArgs++)
		numOutput = atom_getint(argv++);

	if(argc > countArgs++)
  	{
		int hidden;
			
		hidden = atom_getint(argv++);
		// the input and output layers aren't hidden, so numLayers+2 is the total number of layers
		numLayers = hidden+2;
	
		neuronsPerLayer = (unsigned int *)getbytes(numLayers*sizeof(unsigned int));
	
		// number of neurons for the input layer
		neuronsPerLayer[0] = numInput;
	
		// make standard initialization of 3 neurons per hidden layer (if there are too few init args)
		for(i=1; i<=hidden; i++)
		  neuronsPerLayer[i] = 3;
	
		// now check init args
		// arguments 3+: arbitrarily-sized list specifying each hidden layer's number of neurons
		for(i=1; i<=hidden; i++)
		{
			if(argc > countArgs++)
				neuronsPerLayer[i] = atom_getint(argv++);
		}
	
		// number of neurons for the output layer
		neuronsPerLayer[numLayers-1] = numOutput;
	
		activated = true;
	}
	
	// penultimate argument: connection rate
	if(argc > countArgs++)
		connectionRate = atom_getfloat(argv++);

	// final argument: learning rate
	if(argc > countArgs++)
		learningRate = atom_getfloat(argv++);

	// make one hidden layer with 3 neurons as standard, if there were too few init args
	if(!activated)
	{
		neuronsPerLayer = (unsigned int *)getbytes(3*sizeof(unsigned int));
		neuronsPerLayer[0] = numInput;
		neuronsPerLayer[1] = 3;
		neuronsPerLayer[2] = numOutput;
	}

	// ... end of parsing init args
  
  
	if(x->x_ann)
		pdFann_free(x);
  
  
	//WB: this old line from [ann_mlp] (which used FANN 1.0) should now be fann_create_sparse_array() in FANN 2.0, but there is no argument for learningRate
	// x->x_ann = fann_create_array(connectionRate, learningRate, numLayers, neuronsPerLayer);
	x->x_ann = fann_create_sparse_array(connectionRate, numLayers, neuronsPerLayer);

	// deallocate helper array
	t_freebytes(neuronsPerLayer, numLayers * sizeof(unsigned int));
  
	if(!x->x_ann)
	{
		pd_error(x, "%s: error creating the ANN", x->x_objSymbol->s_name);
		return;
	}
  
	pdFann_allocateStorage(x);

	// WB: have to set this explicitly afterward because fann_create_sparse_array() doesn't take a learning rate argument like fann_create_array used to in FANN 1.0
	fann_set_learning_rate(x->x_ann, learningRate);
	fann_set_activation_function_hidden(x->x_ann, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_function_output(x->x_ann, FANN_SIGMOID);

	// store the pointer to dataspace in fann struct's user_data element
	// this allows us to access elements of x in the custom callback function
	x->x_ann->user_data = (void *)x;

	// must set errdat argument to NULL, and could pass an opened file pointer instead of stderr in order to print errors to a text file. Not bothering with this now
//	fann_set_error_log(NULL, stderr);

	// change the callback from the default to our custom function above to print and/or send data out object outlets
	fann_set_callback(x->x_ann, ann_custom_callback);

	pdFann_print(x);
}


static void pdFann_destroyFann(t_fann *x)
{
	if(!x->x_ann)
	{
		pd_error(x, "%s: ANN not initialized", x->x_objSymbol->s_name);
		return;
	}
	else
	{
		fann_destroy(x->x_ann);
		x->x_ann = NULL;
		post("%s: destroyed ANN", x->x_objSymbol->s_name);
	}
}


static void pdFann_status(t_fann *x)
{
	if(x->x_mode==TRAIN)
		post("%s: train mode", x->x_objSymbol->s_name);
	else
		post("%s: run mode", x->x_objSymbol->s_name);
}


static void pdFann_train(t_fann *x)
{
	x->x_mode=TRAIN;
	
	if(!x->x_ann)
	{
		pd_error(x, "%s: ANN not initialized", x->x_objSymbol->s_name);
		return;
	}
	
	fann_reset_MSE(x->x_ann);
	// TODO: should we randomize weights here as well?
	pdFann_status(x);
}


static void pdFann_run(t_fann *x)
{
	x->x_mode=RUN;
	pdFann_status(x);
}


static void pdFann_setPostCallback(t_fann *x, t_float flag)
{
	if(flag<=0)
		x->x_postCallback = false;
	else
		x->x_postCallback = true;
	
	if(x->x_postCallback)
		post("%s: posting callback results to Pd console", x->x_objSymbol->s_name);
	else
		post("%s: suppressing callback posts to Pd console", x->x_objSymbol->s_name);
}


static void pdFann_trainOnFile(t_fann *x, t_symbol *s)
{
	// make correct path
	char patcherPath[MAXPDSTRING];
	char fileName[MAXPDSTRING];

	if(!x->x_ann)
	{
		pd_error(x, "%s: ANN not initialized", x->x_objSymbol->s_name);
		return;
	}

	// make correct path
	canvas_makefilename(x->x_canvas, s->s_name, patcherPath, MAXPDSTRING);
	sys_bashfilename(patcherPath, fileName);
	x->x_fileNameTrain = gensym(fileName);

	if(!x->x_fileNameTrain)
	{
		pd_error(x, "%s: failed to create filename as argument to fann_train_on_file", x->x_objSymbol->s_name);
		return;
  	}
  	
	post("%s: starting training on file %s", x->x_objSymbol->s_name, x->x_fileNameTrain->s_name);

	fann_train_on_file(x->x_ann, x->x_fileNameTrain->s_name, x->x_maxIterations, x->x_iterationsBetweenReports, x->x_desiredError);
	
	post("%s: finished training on file %s", x->x_objSymbol->s_name, x->x_fileNameTrain->s_name);
}


static void pdFann_setDesiredError(t_fann *x, t_float desiredError)
{	
	if(desiredError>0.0)
	{
		x->x_desiredError = desiredError;
		post("%s: desired_error set to %f", x->x_objSymbol->s_name, x->x_desiredError);
	}
	else
		pd_error(x, "%s: desired_error must be greater than zero", x->x_objSymbol->s_name);
}


static void pdFann_setMaxIterations(t_fann *x, t_float maxIterations)
{	
	if(maxIterations>0)
	{
		x->x_maxIterations = maxIterations;
		post("%s: max_iterations set to %i", x->x_objSymbol->s_name, x->x_maxIterations);
	}
	else
		pd_error(x, "%s: max_iterations must be greater than zero", x->x_objSymbol->s_name);
}


static void pdFann_setIterationsBetweenReports(t_fann *x, t_float iterationsBetweenReports)
{
	if(iterationsBetweenReports>0)
	{
		x->x_iterationsBetweenReports = iterationsBetweenReports;
		post("%s: iterations_between_reports set to %i", x->x_objSymbol->s_name, x->x_iterationsBetweenReports);
	}
	else
		pd_error(x, "%s: iterations_between_reports must be greater than zero", x->x_objSymbol->s_name);
}


// run the ann using floats in list passed to the inlet as input values
// and send result to outlet as list of floats
static void pdFann_runTheNet(t_fann *x, t_symbol *s, unsigned int argc, t_atom *argv)
{
	unsigned int i, numInput, numOutput;	
	fann_type *predictions;
	// need to use a local fann_type pointer to take return value of fann_run() below
	// it can't be previously allocated as a specific size, as x->x_outputFann is
	// see fann.c/fann_run() and fann_data.h for more detail

	if(!x->x_ann)
	{
		pd_error(x, "%s, ANN not initialized", x->x_objSymbol->s_name);
		return;
	}

	numInput = fann_get_num_input(x->x_ann);
	numOutput = fann_get_num_output(x->x_ann);

	if(argc < numInput)
	{
		pd_error(x, "%s: input list does not contain enough elements", x->x_objSymbol->s_name);
		return;
	}

	// fill input array with actual data sent to inlet
	for (i=0;i<numInput;i++)
		x->x_input[i] = atom_getfloat(argv++);
	
	// run the ann
	predictions = fann_run(x->x_ann, x->x_input);

	// fill the output array with result from ann
	for (i=0;i<numOutput;i++)
		SETFLOAT(x->x_outputAtom+i, predictions[i]);

	// send output array to outlet
	outlet_list(x->x_listOut, 0, numOutput, x->x_outputAtom);
}


// WB: TODO: this should have a check to only allow training type FANN_TRAIN_INCREMENTAL
static void pdFann_trainOnTheFly(t_fann *x, t_symbol *s, int argc, t_atom *argv)
{
	unsigned int i, numInput, numOutput;
	t_float mse;

	if(!x->x_ann)
	{
		pd_error(x, "%s, ANN not initialized", x->x_objSymbol->s_name);
		return;
	}

	numInput = fann_get_num_input(x->x_ann);
	numOutput = fann_get_num_output(x->x_ann);

	if((int)(numInput + numOutput)!= argc)
	{
		pd_error(x, "%s: incorrect number of arguments passed. In training mode you must pass a list with (num_input + num_output) floats", x->x_objSymbol->s_name);
		return;
	}

	// fill input array with actual data sent to inlet
	for (i=0; i<numInput; i++)
		x->x_input[i] = atom_getfloat(argv++);

	for (i=0; i<numOutput; i++)
		x->x_outputFann[i] = atom_getfloat(argv++);
	
	fann_train(x->x_ann, x->x_input, x->x_outputFann);

	mse = fann_get_MSE(x->x_ann);
	
	outlet_float(x->x_mseOut, mse);
}


static void pdFann_manageList(t_fann *x, t_symbol *s, int argc, t_atom *argv)
{
	if(x->x_mode)
		pdFann_runTheNet(x, s, argc, argv);
	else
		pdFann_trainOnTheFly(x, s, argc, argv);
}


static void pdFann_setFilename(t_fann *x, t_symbol *s)
{
	// make correct path
	char patcherPath[MAXPDSTRING];
	char fileName[MAXPDSTRING];
  
	if(!s)
	{
		pd_error(x, "%s: no file name given", x->x_objSymbol->s_name);
		return;
  	}
  	
	// make correct path
	canvas_makefilename(x->x_canvas, s->s_name, patcherPath, MAXPDSTRING);
	sys_bashfilename(patcherPath, fileName);
	x->x_fileName = gensym(fileName);
}


static void pdFann_loadAnnFromFile(t_fann *x, t_symbol *s)
{
	pdFann_setFilename(x, s);
  
	if(!x->x_fileName)
	{
		pd_error(x, "%s: no file name given", x->x_objSymbol->s_name);
		return;
	}
  
	// deallocate storage
	if(x->x_ann)
		pdFann_free(x);
  
	x->x_ann = fann_create_from_file(x->x_fileName->s_name);

	if(!x->x_ann)
		pd_error(x, "%s: error opening %s", x->x_objSymbol->s_name, x->x_fileName->s_name);
	else
		post("%s: ANN loaded fom file %s", x->x_objSymbol->s_name, x->x_fileName->s_name);

	// allocate storage
	pdFann_allocateStorage(x);
}


static void pdFann_saveAnnToFile(t_fann *x, t_symbol *s)
{
	pdFann_setFilename(x, s);

	if(!x->x_fileName)
	{
		pd_error(x, "%s: no file name given", x->x_objSymbol->s_name);
		return;
	}

	if(!x->x_ann)
		pd_error(x, "%s: ANN is not initialized", x->x_objSymbol->s_name);
	else
	{
		fann_save(x->x_ann, x->x_fileName->s_name);
		post("%s: ANN saved in file %s", x->x_objSymbol->s_name, x->x_fileName->s_name);
	}
}


// WB: TODO: this shouldn't be A_GIMME, although the argc is useful
static void pdFann_setTrainingAlgo(t_fann *x, t_symbol *s, int argc, t_atom *argv)
{
	if(!x->x_ann)
	{
		pd_error(x, "%s: ANN is not initialized", x->x_objSymbol->s_name);
		return;
	}

	if(argc>0)
	{
		int algoIdx;
		t_symbol *algoSymbol;

		algoSymbol = atom_getsymbol(argv);
		
		for(algoIdx=0; algoIdx<NUMTRAININGALGOS; algoIdx++)
		{
			if(strcmp(algoSymbol->s_name, FANN_TRAIN_NAMES[algoIdx])==0)
				break;
    	}

		algoIdx = (algoIdx>=NUMTRAININGALGOS)?NUMTRAININGALGOS-1:algoIdx;
		
		fann_set_training_algorithm(x->x_ann, algoIdx);

		post("%s: training algorithm set to %s (%i)", x->x_objSymbol->s_name, 	algoSymbol->s_name, algoIdx);
	}
	else
		pd_error(x, "%s: you must specify the training algorithm", x->x_objSymbol->s_name);
}


// WB: TODO: this shouldn't be A_GIMME
static void pdFann_setActivationFunction(t_fann *x, t_symbol *s, int argc, t_atom *argv)
{
	if(!x->x_ann)
	{
		pd_error(x, "%s: ANN is not initialized", x->x_objSymbol->s_name);
		return;
	}

	if(argc>2)
	{
		unsigned int i, layerIdx, neuronIdx, funcIdx, numLayers, *neuronsPerLayer;
		t_symbol *funcSymbol;

		funcSymbol = atom_getsymbol(argv++);
		layerIdx = atom_getint(argv++);
		neuronIdx = atom_getint(argv);

		numLayers = fann_get_num_layers(x->x_ann);

		if(layerIdx>=numLayers)
		{
			pd_error(x, "%s: layer %i does not exist", x->x_objSymbol->s_name, layerIdx);
			return;
		}
		
		neuronsPerLayer = (unsigned int *)getbytes(numLayers*sizeof(unsigned int));

		// init to zero before getting neurons per layer
		for(i=0; i<numLayers; i++)
			neuronsPerLayer[i] = 0;

		fann_get_layer_array(x->x_ann, neuronsPerLayer);
		
		if(neuronIdx>=neuronsPerLayer[layerIdx])
		{
			pd_error(x, "%s: layer %i, neuron %i does not exist", x->x_objSymbol->s_name, layerIdx, neuronIdx);
			return;
		}
		
		// free the neuronsPerLayer memory
		t_freebytes(neuronsPerLayer, numLayers*sizeof(unsigned int));

		for(funcIdx=0; funcIdx<NUMACTIVATIONFUNCS; funcIdx++)
		{
			if(strcmp(funcSymbol->s_name, FANN_ACTIVATIONFUNC_NAMES[funcIdx])==0)
				break;
    	}

		funcIdx = (funcIdx>=NUMACTIVATIONFUNCS)?NUMACTIVATIONFUNCS-1:funcIdx;
		
		fann_set_activation_function(x->x_ann, funcIdx, layerIdx, neuronIdx);

		post("%s: layer %i, neuron %i activation function set to %s (%i)", x->x_objSymbol->s_name, layerIdx, neuronIdx, funcSymbol->s_name, funcIdx);
	}
	else
		pd_error(x, "%s: you must specify the activation function, layer, and neuron", x->x_objSymbol->s_name);
}


// WB: TODO: this shouldn't be A_GIMME
static void pdFann_setActivationFunctionLayer(t_fann *x, t_symbol *s, int argc, t_atom *argv)
{
	if(!x->x_ann)
	{
		pd_error(x, "%s: ANN is not initialized", x->x_objSymbol->s_name);
		return;
	}

	if(argc>1)
	{
		unsigned int layerIdx, funcIdx;
		t_symbol *funcSymbol;

		funcSymbol = atom_getsymbol(argv++);
		layerIdx = atom_getint(argv);
		
		if(layerIdx>=fann_get_num_layers(x->x_ann))
		{
			pd_error(x, "%s: layer %i does not exist", x->x_objSymbol->s_name, layerIdx);
			return;
		}

		for(funcIdx=0; funcIdx<NUMACTIVATIONFUNCS; funcIdx++)
		{
			if(strcmp(funcSymbol->s_name, FANN_ACTIVATIONFUNC_NAMES[funcIdx])==0)
				break;
    	}

		funcIdx = (funcIdx>=NUMACTIVATIONFUNCS)?NUMACTIVATIONFUNCS-1:funcIdx;
		
		fann_set_activation_function_layer(x->x_ann, funcIdx, layerIdx);

		post("%s: layer %i activation function set to %s (%i)", x->x_objSymbol->s_name, layerIdx, funcSymbol->s_name, funcIdx);
	}
	else
		pd_error(x, "%s: you must specify the activation function and layer", x->x_objSymbol->s_name);
}


// WB: TODO: this shouldn't be A_GIMME
static void pdFann_setActivationFunctionHidden(t_fann *x, t_symbol *s, int argc, t_atom *argv)
{
	if(!x->x_ann)
	{
		pd_error(x, "%s: ANN is not initialized", x->x_objSymbol->s_name);
		return;
	}

	if(argc>0)
	{
		int funcIdx;
		t_symbol *funcSymbol;

		funcSymbol = atom_getsymbol(argv);
		
		for(funcIdx=0; funcIdx<NUMACTIVATIONFUNCS; funcIdx++)
		{
			if(strcmp(funcSymbol->s_name, FANN_ACTIVATIONFUNC_NAMES[funcIdx])==0)
				break;
    	}

		funcIdx = (funcIdx>=NUMACTIVATIONFUNCS)?NUMACTIVATIONFUNCS-1:funcIdx;
		
		fann_set_activation_function_hidden(x->x_ann, funcIdx);

		post("%s: hidden activation function set to %s (%i)", x->x_objSymbol->s_name, 	funcSymbol->s_name, funcIdx);
	}
	else
		pd_error(x, "%s: you must specify the activation function", x->x_objSymbol->s_name);
}


// WB: TODO: this shouldn't be A_GIMME
static void pdFann_setActivationFunctionOutput(t_fann *x, t_symbol *s, int argc, t_atom *argv)
{
	if(!x->x_ann)
	{
		pd_error(x, "%s: ANN is not initialized", x->x_objSymbol->s_name);
		return;
	}

	if(argc>0)
	{
		int funcIdx;
		t_symbol *funcSymbol;

		funcSymbol = atom_getsymbol(argv);
		
		for(funcIdx=0; funcIdx<NUMACTIVATIONFUNCS; funcIdx++)
		{
			if(strcmp(funcSymbol->s_name, FANN_ACTIVATIONFUNC_NAMES[funcIdx])==0)
				break;
    	}

		funcIdx = (funcIdx>=NUMACTIVATIONFUNCS)?NUMACTIVATIONFUNCS-1:funcIdx;
		
		fann_set_activation_function_output(x->x_ann, funcIdx);

		post("%s: output activation function set to %s (%i)", x->x_objSymbol->s_name, funcSymbol->s_name, funcIdx);
	}
	else
		pd_error(x, "%s: you must specify the activation function", x->x_objSymbol->s_name);
}


static void pdFann_setTrainErrorFunction(t_fann *x, t_symbol *s, int argc, t_atom *argv)
{
	if(!x->x_ann)
	{
		pd_error(x, "%s: ANN is not initialized", x->x_objSymbol->s_name);
		return;
	}

	if(argc>0)
	{
		int funcIdx;
		t_symbol *funcSymbol;

		funcSymbol = atom_getsymbol(argv);
	
		for(funcIdx=0; funcIdx<NUMTRAINERRORFUNCS; funcIdx++)
		{
			if(strcmp(funcSymbol->s_name, FANN_ERRORFUNC_NAMES[funcIdx])==0)
				break;
		}

		funcIdx = (funcIdx>=NUMTRAINERRORFUNCS)?NUMTRAINERRORFUNCS-1:funcIdx;
	
		fann_set_train_error_function(x->x_ann, funcIdx);

		post("%s: training error function set to %s (%i)", x->x_objSymbol->s_name, funcSymbol->s_name, funcIdx);
	}
	else
		pd_error(x, "%s: you must specify the training error function", x->x_objSymbol->s_name);
}


static void pdFann_setTrainStopFunction(t_fann *x, t_symbol *s, int argc, t_atom *argv)
{
	if(!x->x_ann)
	{
		pd_error(x, "%s: ANN is not initialized", x->x_objSymbol->s_name);
		return;
	}

	if(argc>0)
	{
		int funcIdx;
		t_symbol *funcSymbol;

		funcSymbol = atom_getsymbol(argv);
	
		for(funcIdx=0; funcIdx<NUMTRAINSTOPFUNCS; funcIdx++)
		{
			if(strcmp(funcSymbol->s_name, FANN_STOPFUNC_NAMES[funcIdx])==0)
				break;
		}

		funcIdx = (funcIdx>=NUMTRAINSTOPFUNCS)?NUMTRAINSTOPFUNCS-1:funcIdx;
	
		fann_set_train_stop_function(x->x_ann, funcIdx);

		post("%s: training stop function set to %s (%i)", x->x_objSymbol->s_name, funcSymbol->s_name, funcIdx);
	}
	else
		pd_error(x, "%s: you must specify the training stop function", x->x_objSymbol->s_name);
}


static void pdFann_randomizeWeights(t_fann *x, t_float min, t_float max)
{
	if(!x->x_ann)
	{
		pd_error(x, "%s: ANN is not initialized", x->x_objSymbol->s_name);
		return;
	}

	fann_randomize_weights(x->x_ann, min, max);
	post("%s: weights randomized between %f and %f", x->x_objSymbol->s_name, min, max);
}


static void pdFann_resetMSE(t_fann *x)
{
	if(!x->x_ann)
	{
		pd_error(x, "%s: ANN is not initialized", x->x_objSymbol->s_name);
		return;
	}
	else
		fann_reset_MSE(x->x_ann);
}


static void pdFann_getLastError(t_fann *x)
{
	//t_symbol *error;

	// in fann_error.c, it looks like the fann_error() function never writes the errno or errstr to x->x_ann. this is because all calls to fann_error() give NULL as the first argument. see fann_train_data.c, for an example. Could change the FANN source to fix this, but that's going too far, and FANN may be fixed in the future. May just abandon getting errors into Pd for now. It's always possible to see them by running Pd from terminal so that the default error routing to stderr will show up in the terminal.

	//error = fann_get_errstr((struct fann_error *)(x->x_ann));
	//post("%s: last error: %s", x->x_objSymbol->s_name, error->s_name);

	//fann_print_error((struct fann_error *)(x->x_ann));

	post("%s: method not yet implemented", x->x_objSymbol->s_name);
}


static void pdFann_learningRate(t_fann *x, t_float lr)
{
	t_float learnRate;
	
	learnRate = 0.0;

	if(!x->x_ann)
	{
		pd_error(x, "%s: ANN is not initialized", x->x_objSymbol->s_name);
		return;
	}

	learnRate = (lr<0.0) ? 0.0 : lr;
	fann_set_learning_rate(x->x_ann, learnRate);

	post("%s: learning rate set to %f", x->x_objSymbol->s_name, learnRate);

}


static void pdFann_setActivationSteepnessHidden(t_fann *x, t_float ash)
{
	if(!x->x_ann)
	{
		pd_error(x, "%s: ANN is not initialized", x->x_objSymbol->s_name);
		return;
	}

	fann_set_activation_steepness_hidden(x->x_ann, ash);
}


static void pdFann_setActivationSteepnessOutput(t_fann *x, t_float aso)
{
	if(!x->x_ann)
	{
		pd_error(x, "%s: ANN is not initialized", x->x_objSymbol->s_name);
		return;
	}

	fann_set_activation_steepness_output(x->x_ann, aso);
}


static void *pdFann_new(t_symbol *s, int argc, t_atom *argv)
{
	t_fann *x = (t_fann *)pd_new(fann_class);
	x->x_listOut = outlet_new(&x->x_obj, &s_list);
	x->x_epochsOut = outlet_new(&x->x_obj, &s_float);
	x->x_mseOut = outlet_new(&x->x_obj, &s_float);

	// store the pointer to the symbol containing the object name. Can access it for error and post functions via s->s_name
	x->x_objSymbol = s;
	
	x->x_desiredError = 0.001;
	x->x_maxIterations = 500000;
	x->x_iterationsBetweenReports = 1000;
	x->x_mode=RUN;
	x->x_canvas = canvas_getcurrent();
	x->x_fileName = NULL;
	x->x_fileNameTrain = NULL;
	x->x_ann = NULL;
	x->x_input = NULL;
	x->x_outputAtom = NULL;
	x->x_outputFann = NULL;
	x->x_postCallback = false;
	
	if(argc>0)
	{
		x->x_fileName = atom_gensym(argv);
		pdFann_loadAnnFromFile(x, NULL);
	}

	return (void *)x;
}


void fann_setup(void)
{
	post("[fann]: multilayer perceptron for Pd");
	post("version: "VERSION"");
	post("compiled: "__DATE__);
	post("author: Davide Morelli");
	post("updated for FANN 2.0 by: William Brent, 2018");

	fann_class = class_new(
		gensym("fann"),
		(t_newmethod)pdFann_new,
		(t_method)pdFann_free, sizeof(t_fann),
		CLASS_DEFAULT,
		A_GIMME,
		0
	);
	
	class_addmethod(
		fann_class,
		(t_method)pdFann_createFann,
		gensym("create"),
		A_GIMME,
		0
	);

	class_addmethod(
		fann_class,
		(t_method)pdFann_destroyFann,
		gensym("destroy"),
		0
	);

	class_addmethod(
		fann_class,
		(t_method)pdFann_train,
		gensym("train"),
		0
	);
	
	class_addmethod(
		fann_class,
		(t_method)pdFann_run,
		gensym("run"),
		0
	);
	
	class_addmethod(
		fann_class,
		(t_method)pdFann_trainOnFile,
		gensym("train_on_file"),
		A_DEFSYMBOL,
		0
	);
	
	class_addmethod(
		fann_class,
		(t_method)pdFann_loadAnnFromFile,
		gensym("load"),
		A_DEFSYMBOL,
		0
	);
	
	class_addmethod(
		fann_class,
		(t_method)pdFann_saveAnnToFile,
		gensym("save"),
		A_DEFSYMBOL,
		0
	);

	class_addmethod(
		fann_class,
		(t_method)pdFann_print,
		gensym("print"),
		0
	);
	
	class_addmethod(
		fann_class,
		(t_method)pdFann_printParameters,
		gensym("print_parameters"),
		0
	);
	
	// change training parameters
	class_addmethod(
		fann_class,
		(t_method)pdFann_setDesiredError,
		gensym("desired_error"),
		A_DEFFLOAT,
		0
	);
	
	class_addmethod(
		fann_class,
		(t_method)pdFann_setMaxIterations,
		gensym("max_iterations"),
		A_DEFFLOAT,
		0
	);
	
	class_addmethod(
		fann_class,
		(t_method)pdFann_setIterationsBetweenReports,
		gensym("iterations_between_reports"),
		A_DEFFLOAT,
		0
	);
	
	class_addmethod(
		fann_class,
		(t_method)pdFann_learningRate,
		gensym("learning_rate"),
		A_DEFFLOAT,
		0
	);

	// change training  and activation algorithms
	class_addmethod(
		fann_class,
		(t_method)pdFann_setTrainingAlgo,
		gensym("set_training_algo"),
		A_GIMME,
		0
	);

	class_addmethod(
		fann_class,
		(t_method)pdFann_setActivationFunction,
		gensym("set_activation_function"),
		A_GIMME,
		0
	);
	
	class_addmethod(
		fann_class,
		(t_method)pdFann_setActivationFunctionLayer,
		gensym("set_activation_function_layer"),
		A_GIMME,
		0
	);
	
	class_addmethod(
		fann_class,
		(t_method)pdFann_setActivationFunctionOutput,
		gensym("set_activation_function_output"),
		A_GIMME,
		0
	);
	
	class_addmethod(
		fann_class,
		(t_method)pdFann_setActivationFunctionHidden,
		gensym("set_activation_function_hidden"),
		A_GIMME,
		0
	);

	class_addmethod(
		fann_class,
		(t_method)pdFann_setTrainErrorFunction,
		gensym("set_train_error_function"),
		A_GIMME,
		0
	);

	class_addmethod(
		fann_class,
		(t_method)pdFann_setTrainStopFunction,
		gensym("set_train_stop_function"),
		A_GIMME,
		0
	);

	class_addmethod(
		fann_class,
		(t_method)pdFann_setActivationSteepnessHidden,
		gensym("set_activation_steepness_hidden"),
		A_DEFFLOAT,
		0
	);
	
	class_addmethod(
		fann_class,
		(t_method)pdFann_setActivationSteepnessOutput,
		gensym("set_activation_steepness_output"),
		A_DEFFLOAT,
		0
	);
	
	class_addmethod(
		fann_class,
		(t_method)pdFann_setPostCallback,
		gensym("post_callback"),
		A_DEFFLOAT,
		0
	);
	
	// initialization:
	class_addmethod(
		fann_class,
		(t_method)pdFann_randomizeWeights,
		gensym("randomize_weights"),
		A_DEFFLOAT,
		A_DEFFLOAT,
		0
	);

	class_addmethod(
		fann_class,
		(t_method)pdFann_resetMSE, gensym("reset_mse"),
		0
	);

	class_addmethod(
		fann_class,
		(t_method)pdFann_getLastError, gensym("last_error"),
		0
	);
	
	// the most important one: running the ann
	class_addlist(
		fann_class,
		(t_method)pdFann_manageList
	);
}