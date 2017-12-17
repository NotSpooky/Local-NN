module dense;

import activations;
import std.functional : unaryFun;
import std.algorithm;

struct Dense (int neurons, int neuronsLayerBefore, DataType = float, alias activation = Linear!DataType) {
    alias WeightVector = DataType [neuronsLayerBefore][neurons];
    alias OutVector = DataType [neurons];
    alias InVector = DataType [neuronsLayerBefore];
    WeightVector weights;
    OutVector biases;
    
    this (T) (T weightInitialization) {
        foreach (ref neuronWeights; weights) {
            foreach (ref weight; neuronWeights) {
                weight = weightInitialization;
            }
        }
        foreach (ref bias; biases) {
            bias = weightInitialization;
        }
    }


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
    auto backprop (alias updateFunction) (
        in OutVector errorVector,
        in InVector activationVector
    ) {
        // Implementation note: Weights and activations should be updated
        // simultaneously.
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
            foreach (j, weight; weights [neuronPos]) {
                activationErrors [j] += effectInError * weight;
                auto weightDerivative = effectInError * activationVector [j];
                weight -= changeBasedOn (weightDerivative);
            }
        }
        /+debug {
            import std.stdio;
            writeln (`Biases after: `, biases);
            writeln (`Weights after: `, weights);
            writeln (`Activation errors: `, activationErrors);
        }+/
        // TODO: Check if returning this is UB
        return activationErrors;
    }
}

auto getValue (alias value) () {
    import std.traits : isCallable;
    static if (isCallable!value) {
        return value ();
    } else {
        return value;
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
auto neuralNetwork (int inputLen, Layer [] layers, DataType = float, T)(T weightInitialization) {
    class NN {
        import std.conv : text, to;
        import std.range;
        mixin (nnGenerator(inputLen, layers));

        this () {
            static foreach (i, layer; layers) {
                // Call the constructors of all the layers.
                mixin (text (`layer`, i, ` = typeof (layer`, i, `)(getValue!weightInitialization);`));
            }
        }
        
        void train (alias optimizer, alias errorFun, bool printError = true, R1, R2) (int epochs, int batchSize, R1 inputs, R2 labels) {
            assert (inputs.length == labels.length);
            assert (inputs.front.length == inputLen, `incorrect input length for training.`);
            static assert (is (typeof (inputs.front.front) == DataType), 
                text (`Incorrect data type for input, should be `, DataType.stringof));
            static assert (is (typeof (labels.front.front) == DataType), 
                text (`Incorrect data type for labels, should be `, DataType.stringof));

            assert (labels.front.length == outputLen, `incorrect output length for training`);
            foreach (epoch; 0..epochs) {
                import std.random : randomShuffle;
                auto indices = iota (inputs.length).array;
                // Dataset is shuffled each epoch, TODO: Make optional.
                indices.randomShuffle;
                auto activationV = new DataType [totalNeurons];
                foreach (batch; indices.chunks (batchSize)) {
                    activationV [] = 0;
                    auto dataChunks  = inputs.indexed (batch);
                    auto labelChunks = labels.indexed (batch);
                    DataType [inputLen] averageInputs = 0;
                    DataType [outputLen] averageOutputError = 0;
                    foreach (example, label; zip (dataChunks, labelChunks)) {
                        averageInputs [] += example [];
                        this.forward!true (example.to!(DataType [inputLen]), activationV);
                        averageOutputError [] += activationV [$ - outputLen .. $] - label []; 
                    }
                    averageInputs [] /= batch.length;
                    averageOutputError [] /= batch.length;
                    DataType [] lastError = averageOutputError;
                    static if (printError) {
                        import std.stdio;
                        writeln (`lastError = `, lastError.sum);
                    }
                    mixin (mixBackprop ());
                }
            }
        } // End of train
        static string mixBackprop () {
            import std.array : Appender;
            Appender!string toReturn;
            uint [] posInActivationV = [0];
            uint lastActivation = 0;
            foreach (layer; layers [0..$-1]) {
                lastActivation += layer.neurons;
                posInActivationV ~= lastActivation;
            }
            
            assert (posInActivationV.length >= 2);
            foreach_reverse (i, layer; layers) {
                // The activation positions of this layer.
                string error, activationBefore;

                // The activation positions of the layer before.
                if (posInActivationV.length > 1) {
                    auto endPos = posInActivationV.back;
                    auto startPos = posInActivationV [$ - 2];
                    // Has layers before.
                    activationBefore = text (
                        `activationV [`, startPos, `..`, endPos, "]",
                    );
                } else {
                    // Is first layer.
                    activationBefore = `averageInputs`;
                }
                //Eg lastError = 
                //  layer2.backprop!optimizer (
                //     lastError.to!(DataType [2]), activationV [8..24]
                //  );
                string errorIn = text (`lastError.to!(DataType [`, layer.neurons, `])`);
                toReturn ~= text(
                    ` lastError = layer`, i
                    , `.backprop!optimizer (`, errorIn, `, `, activationBefore, ");\n"
                );
                posInActivationV.popBack;
            }
            return toReturn.data;
        }
        //`auto error`, i, ` = layer`, i, `.backprop!optimizer (activationVR []);`
        


    }
    return new NN ();

}

private struct Layer {
    string type;
    uint neurons;
    /// Must be comma-separated and in the order that they are inserted.
    string parameters; 
    
}

private string nnGenerator (int inputLen, Layer [] layers) {
    assert (inputLen > 0);
    assert (layers.length, `Cannot create empty neural network.`);
    import std.array : Appender;
    import std.conv : text;

    uint totalNeurons = layers.map!(a => a.neurons).sum;

    Appender!string toReturn;

    // Useful constants
    toReturn ~= text (
          `enum totalNeurons = `, totalNeurons, ";\n"
        , `enum inputLen = `, inputLen, ";\n"
        , `enum outputLen = `, layers [$-1].neurons, ";\n"
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
            , `!(`
                , layer.neurons, `, `
                , layerInputLen
                , `, DataType, `
                , layer.parameters
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
    // TODO: Split in a version that uses the activation vector and other that doesn't.
    "auto forward (bool training = false)(DataType [inputLen] input, ref DataType [] activationVector) {\n"
        , "\tassert (activationVector || !training, `To train need to receive the activation vector`);\n"
        , "\tassert (!activationVector || activationVector.length >= totalNeurons, `Activation should be null if training = false, and have enough space otherwise.`);\n"
        , "\t", buffers
        );
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
            // Eg. activationVector [16..32] = buffers [0][0..16];
            , "\tstatic if (training)\n"
                , "\t\tactivationVector [", endOfLastActivation, `..`, endOfLastActivation + layer.neurons, `] = `, layerOutput, ";\n");
            endOfLastActivation += layer.neurons;
        }
        
        toReturn ~= text(
            "\tstatic if (training)\n"
            // Could be optimized to receive it by parameter.
                , "\t\treturn activationVector.dup;\n"
            , "\telse\n"
            // If not training, return the last used buffer.
                , "\t\treturn buffers [" ,(layers.length + 1) % 2, `]`
                , "[0..outputLen].dup;\n",
        "}\n");
        // End of forward.

        
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
        ) (0.5); // Can use both a function or a value as weight initialization.
        //writeln (*a);
        float [a.totalNeurons] activationV = 0;
        auto activationVR = activationV [];
        float [] noArr;
        a.forward ([1,2,3,4], noArr);
        a.forward!true ([1,2,3,4], activationVR);
        a.train !(`a/30`, `(a - b)*(a - b)`)(199, 2, [[1f,2,3,4],[2f,3,4,5],[3f,4,5,6]], [[4f,3], [5f,4], [6f,5]]);
        a.forward ([1,2,3,4], noArr).writeln;
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
