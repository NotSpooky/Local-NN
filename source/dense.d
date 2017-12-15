module dense;

import activations;

struct Dense (int neurons, int neuronsLayerBefore, DataType = float, alias activation = Linear!DataType) {
    alias WeightVector = DataType [neuronsLayerBefore][neurons];
    alias OutVector = DataType [neurons];
    alias InVector = DataType [neuronsLayerBefore];
    WeightVector weights;
    OutVector biases;
    void forward (in InVector lastLayerActivations, out OutVector ret) {
        foreach (i; 0..neurons) {
            import std.numeric : dotProduct;
            ret [i] = activation (dotProduct (lastLayerActivations, weights [i]) + biases [i]);
        }
    }
    // errorVector contains the expected change in the outputs of this layer.
    // To determine how much to change for each neuron, the following algorithm
    // is used:
    // TODO: Might be useful to use doubles instead of floats for backprop.
    import std.functional : unaryFun;
    auto backprop (alias updateFunction) (
        in OutVector errorVector,
        in InVector activations
    ) {

        /+debug {
            import std.stdio;
            writeln (`Biases before: `, biases);
            writeln (`Weights before: `, weights);
        }+/
        alias changeBasedOn = unaryFun!updateFunction;
        DataType [neuronsLayerBefore] activationErrors = 0;
        foreach (neuronPos, error; errorVector) {
            auto effectInError = error * activation.derivative (error);
            biases [neuronPos] -= changeBasedOn (effectInError);
            foreach (j, ref weight; weights [neuronPos]) {
                activationErrors [j] += effectInError * weight;
                auto weightDerivative = effectInError * activations [j];
                weight -= changeBasedOn (weightDerivative);
            }
        }
        /+debug {
            import std.stdio;
            writeln (`Biases after: `, biases);
            writeln (`Weights after: `, weights);
            writeln (`Activation errors: `, activationErrors);
        }+/
        return activationErrors;
    }
}


/+
// TODO: Allow arrays of layers or similar.
/// A succession of numLayers identical dense layers.
struct NeuralNetwork (DataType, uint inputLen, alias activation, Layers ...) {

    static foreach (i, layer; Layers) {
        //auto layer = Layer (DataType.stringof, );
    }
}
+/

/// Expects layers to have the following CT parameter format:
/// neurons, neuronsLayerBefore, DataType, rest of parameters.
// TODO: Change to ... instead of [].
// Layers are named layer0, layer1 ...
auto neuralNetwork (int inputLen, Layer [] layers)() {
    struct NN {
        pragma (msg, nnGenerator (inputLen, layers));
        mixin (nnGenerator(inputLen, layers));
    }
    return new NN ();

}

private struct Layer {
    string type;
    uint neurons;
    /// Must be comma-separated and in the order that they are inserted.
    string parameters; 
    
}

private string nnGenerator (int inputLen, Layer [] layers, string dataType = `float`) {
    assert (inputLen > 0);
    assert (layers.length, `Cannot create empty neural network.`);
    import std.array : Appender;
    import std.conv : text;
    import std.algorithm;

    uint totalNeurons = layers.map!(a => a.neurons).sum;

    Appender!string toReturn;

    // Useful constants
    toReturn ~= text (
          `enum totalNeurons = `, totalNeurons, ";\n"
        , `enum inputLen = `, inputLen, ";\n"
        , `alias DataType = `, dataType, ";\n"
    );

    ////////////////////
    // Layer creation //
    ////////////////////

    foreach (i, layer; layers) {
        auto layerInputLen = i == 0 ? inputLen : layers [i-1].neurons;
        toReturn ~= text (
            // Eg. Dense
            layer.type 
            //Eg. !(3, 4, float, Linear!float)
            , `!(`    , layer.neurons
                , `, `, layerInputLen
                , `, `, dataType
                , `, `, layer.parameters
            ,`)` 
            // Eg. layer3;
            , ` layer`, i , ";\n");
    }

    /////////////////////
    // Buffer creation //
    /////////////////////

    // Used to allocate an array of the appropiate length for the buffers.
    // Used in the forward method.
    // Could be optimized to use the second biggest also for the other buffer.
    uint maxAmountOfNeurons = layers.map!(a => a.neurons).reduce!max;
    // buffers).
    // TODO: Optimize the case where activations needn't to be saved
    // (Just predicting, not training).
    // Eg. float [8][2] buffers;
    string buffers = text (`DataType [`, maxAmountOfNeurons, `]`
        // If there's only 1 layer, no need for 2 buffers.
        ,`[`,layers.length > 1 ? 2 : 1,`]`,
        "buffers;\n" );

    //////////////////////////////
    // Forward method creation  //
    // Prepare for spaguetti ;) //
    //////////////////////////////

    toReturn ~= text (
    // If the training parameter is used, then the activations are returned,
    // the predicted output is returned otherwise.
    "auto forward (bool training = false)(DataType [inputLen] input) {\n"
        , "\t", buffers
        , "\tstatic if (training)\n"
        // Eg. float [150] activations = 0;
            , "\t\tDataType [totalNeurons] activations = 0;\n");
        uint endOfLastActivation = 0;
        // Each layer outputs to alternating buffers.
        // If training is also necessary to output to the activation buffer so that
        // backpropagation knows them.
        foreach (i, layer; layers) {
            string layerInput = i == 0 ? 
                `input` // First layer receives from input.
                // Other layers receive from the output of the last layer.
                : text(`buffers [`, (i + 1) % 2, `][0..`,layers [i-1].neurons,`]`);
            string layerOutput = text (`buffers [`,i % 2, `][0..`, layer.neurons ,`]`);
            toReturn ~= text (
            // Eg. layer0.forward (input, cast (float [8]) buffers [0][0..8]);
            //     layer1.forward (buffers [0][0..8], cast (float [16]) buffers [1][0..16]);
            //     return buffers [1][0..16].dup;
            "\tlayer", i, `.forward (`, layerInput , `, cast (DataType [`, layer.neurons , `])`, layerOutput, ");\n"
            // Eg. activations [16..32] = buffers [0][0..16];
            , "\tstatic if (training)\n"
                , "\t\tactivations [", endOfLastActivation, `..`, endOfLastActivation + layer.neurons, `] = `, layerOutput, ";\n");
            endOfLastActivation += layer.neurons;
        }
        
        // Where is the output of the last layer stored:
        string lastBuffer = text (
            `buffers [` ,(layers.length + 1) % 2, `][0..`, layers [$-1].neurons, `]`
        );

        toReturn ~= text(
            "\tstatic if (training)\n"
            // Could be optimized to receive it by parameter.
                , "\t\treturn activations.dup;\n"
            , "\telse\n"
            // If not training, return the last used buffer.
                , "\t\treturn ", lastBuffer, ".dup;\n",
        "}\n");
        // End of forward.

        //////////////////////////////
        // Training method creation //
        //////////////////////////////

        toReturn ~= text ("void train (R1, R2)(int epochs, int batchSize, R1 inputs, R2 labels) {\n"
        , "\timport std.range;\n"
        , "\tassert (inputs.length == labels.length);\n"
        , "\tassert (inputs.front.length == inputLen, `incorrect input length for training.`);\n"
        , "\tassert (labels.front.length == ", layers [$-1].neurons, ", `incorrect output length for training`);\n"
        , "\tforeach (epoch; 0..epochs) {\n"
        , "\t\timport std.random : randomShuffle;\n"
        , "\t\timport std.conv :to;\n"
        , "\t\timport std.algorithm;\n"
        , "\t\tauto indices = iota (inputs.length).array;\n"
        // Dataset is shuffled each epoch, TODO: Make optional.
        , "\t\tindices.randomShuffle;"
        , "\t\tforeach (batch; indices.chunks (batchSize)){\n"
        , "\t\t\tauto dataChunks  = inputs.indexed (batch);"
        , "\t\t\tauto labelChunks = labels.indexed (batch);"
        , "\t\t\tauto activations = dataChunks
            .map!(a => a.to!(DataType[inputLen]))
            .map!(a => forward!true (a));\n"
        , "\t\t\tforeach (output, label; activations.zip (labelChunks)) {\n"
        , "\n\t\t\timport std.stdio; writeln (output, label);\n"
        , "\t\t\t}\n"
        , "\t\t}\n"
        , "\t}\n"
        , "}\n");
        
    return toReturn.data;
}

unittest {

    //import std.meta : AliasSeq;
    //NeuralNetwork!(float, 2, 2, Linear!float, Dense) nn;
    debug {
        import std.stdio;
        /+
        writeln (nnGenerator (4, [
            Layer (`Dense`, 8, `float, Linear!float`)
            , Layer (`Dense`, 4, `float, Linear!float`)
        ]));
        +/
        auto a = neuralNetwork!(
            /* Input length */ 4,
            [
              Layer(`Dense`, 8)
            , Layer (`Dense`, 16, `ReLU!float`)
            , Layer (`Dense`, 2, `Linear!float`)
            ]
        );
        //writeln (*a);
        a.forward ([1,2,3,4]);
        a.forward!true ([1,2,3,4]);
        a.train (2, 2, [[1f,2,3,4],[2f,3,4,5],[3f,4,5,6]], [[4f,3], [5f,4], [6f,5]]); 
    }

    auto inputLayer = Dense!(4, 2) ();
    inputLayer.weights = [[.1f,.2], [.3,.4], [.5,.6], [.7,.8]];
    inputLayer.biases   = [.1f, .2, .3, .4];
    auto hiddenLayer = Dense!(4, 4) ();
    hiddenLayer.weights = [[.1f,.2,.3,.4],[.1,.2,.3,.4],[.1,.2,.3,.4],[.1,.2,.3,.4]];
    hiddenLayer.biases   = [.1f, .2, .3, .4];
    auto outputLayer = Dense!(2, 4) ();
    outputLayer.weights = [[.1f,.2,.3,.4], [.1f,.2,.3,.4]];
    outputLayer.biases   = [.1f, .2];

    float [2] inputData = [0f, 4f];
    float [2] expectedOutput = [2f, 3];
    float [2] outputLayerOut;

    import std.stdio;
    float [4] inputLayerOut;
    inputLayer.forward (inputData, inputLayerOut);
    //writeln (`inputLayerOut: `, inputLayerOut);

    float [4] hiddenLayerOut;
    hiddenLayer.forward (inputLayerOut, hiddenLayerOut);
    //writeln (`hiddenLayerOut: `, hiddenLayerOut);

    outputLayer.forward (hiddenLayerOut, outputLayerOut);
    //writeln (`outputLayerOut: `, outputLayerOut);

    float [2] error = outputLayerOut [] - expectedOutput [];
    //writeln (`Linear error: `, error);

    auto hiddenErrors = outputLayer.backprop!`a/30`(error, hiddenLayerOut);
    auto inputErrors =  hiddenLayer.backprop!`a/30`(hiddenErrors, inputLayerOut);
    inputLayer.backprop!`a/30` (inputErrors, inputData);
}
