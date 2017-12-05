import std.stdio, std.range;
import std.traits : isCallable;

// Example usage.
void main()
{
    import std.random;
    auto gen = Random(unpredictableSeed); // Random generator.
    // How many inputs does each neuron have.
    enum connectivity = 32;
    enum layers = 128;
    enum neuronsPerLayer = 64;
    auto nn = new LocalNN!(connectivity, layers, neuronsPerLayer,a => a/2f)
        (/* Weight initialization function */ () => uniform (-0.5f, 0.5f, gen));

    // Example input: 64 random floats from 0 to 1.
    float [64] input = uniform (0f, 1f, gen);
    nn.forward (input).writeln;
}

// Implements a locally connected neural network with all the layers of the same
// connectivity and amount of neurons.
// Connectivity = neurons in layer before connected to each single neuron
// in this layer.
struct LocalNN (int connectivity, int layers, int neuronsPerLayer
/**/ , alias activation, DataType = float) {
    static assert (connectivity > 0 && layers > 0 && neuronsPerLayer > 0);
    alias SingleNeuronWeights = DataType [connectivity];
    LocalLayer!(neuronsPerLayer, connectivity, activation, DataType) [layers] nn;
    this (F1)(F1 initFunction) if (isCallable!F1) {
        // Init all the layers with the init function.
        // TODO: Allow F1 to be value instead of function.
        nn = LocalLayer!(neuronsPerLayer, connectivity, activation, DataType)(initFunction);
    }
    auto forward (in DataType [neuronsPerLayer] input, DataType biasInit = 1) {
        // Possible optimization just initializing the bias neurons.
        DataType [neuronsPerLayer + connectivity - 1] buf1 = biasInit;
        DataType [neuronsPerLayer + connectivity - 1] buf2 = biasInit;
        // Used to switch between buffers.
        bool iterationParity = true;
        enum initialNonBias = connectivity / 2;
        enum endNonBias = connectivity / 2 + neuronsPerLayer;
        buf1 [initialNonBias .. endNonBias] = input;

        foreach (layer; nn) {
            // Possible unrolling.
            if (iterationParity) {
                layer.forward (buf1, buf2 [initialNonBias .. endNonBias]);
            } else {
                layer.forward (buf2, buf1 [initialNonBias .. endNonBias]);
            }
            iterationParity ^= true; // Switch.
        }
        return iterationParity ?
            buf1 [initialNonBias .. endNonBias].dup
            : buf2 [initialNonBias .. endNonBias].dup;
    }
}

struct LocalLayer (int neurons, int connectivity, alias activation
/**/ , DataType = float) {
    @disable this ();
    this (R)(R weights) if (isInputRange!R){
        import std.conv : to;
        this.weights = weights.to!Conns;
    }
    this (F)(F initFunction) if (isCallable!F) {
        foreach (ref weight; weights) {
            weight = initFunction ();
        }
    } 
    static assert (neurons > 0 && connectivity > 0);
    private enum totalConnections = neurons * connectivity;
    private enum inputLen = neurons + connectivity - 1;
    alias Conns = DataType [totalConnections];
    Conns weights;
    // TODO: Check if using just input makes it faster.
    void forward (in DataType [inputLen] input, ref DataType [neurons] output) {
        uint weightStart = 0;
        foreach (i; 0..neurons) {
            import std.numeric : dotProduct;
            // TODO: Check if manual optimization needed.
            output [i] = activation (dotProduct (input [i..i+connectivity], weights [weightStart..weightStart + connectivity]));
            weightStart += connectivity;
        }
    }
}

unittest { // Test single layers.
    enum neurons = 8;
    // Each neuron connected to this amount of neurons in the layer before.
    enum connectivity = 2; 
    alias LType = LocalLayer!(neurons, connectivity, linear);
    // Initialize with explicit weights.
    auto layer = LType ([1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6]);
    float [neurons + connectivity -1] input = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    float [neurons] output;
    layer.forward (input, output);
    // Each element should be 1*1 + 2*2, 3*2 + 4*3, 5*3 + 6*4, ...
    assert (output == [5, 18, 39, 68, 45, 20, 53, 94]);
    // Layers with all weights as 1.
    layer = LType (() => 1f);
    layer.forward (input, output);
    // Each element should be 1 + 2, 2 + 3, ...
    assert (output == [3, 5, 7, 9, 11, 13, 15, 17]);
}

unittest { // Test locally connected NN.
    enum neurons = 8;
    enum connectivity = 2;

    // Test single layer.
    auto nn = LocalNN!(connectivity, 1 /* layer */, neurons, linear)(() => 1);
    // Forward with biases initialized with 0.
    auto output = nn.forward ([1, 2, 3, 4, 5, 6, 7, 8], 0f /*Biases at 0*/);
    // Each element should be 0*1 + 1*1, 1*1 + 2*1 + 2*1 + 3*1, ...
    // The first 1 is because there's a bias neuron with 0 before.
    assert (output == [1, 3, 5, 7, 9, 11, 13, 15], `Test single layer failed`);

    // Test 2 layers
    auto nn2 = LocalNN!(connectivity, 2 /* layers */, neurons, linear)(() => 1);
    output = nn2.forward ([1, 2, 3, 4, 5, 6, 7, 8], 0f/*Biases at 0*/);
    // Each element should be the result of nn.forward but joined a level 
    // more:
    // 0*1 (bias) + 1*1, 1*1 + 3*1, 3*1 + 5*1, ...
    assert (output == [1, 4, 8, 12, 16, 20, 24, 28], `Test 2 layers failed`);
}

// Activations.
float linear (float input) {return input;}
int ilinear (int input) {return input;}

// Each neuron in a layer sums the values of the neurons connected to it,
// then applies the activation function and that is sent to the neurons that
// are listening to it.

// In the case of locally connected layers, each neuron receives from the nearest 
// N inputs, to make the algorithm faster, several bias neurons need to exist in
// the layer before (added on the forward function).

// Need to add connectivity - 1 bias neurons to the layer before so that the algorithm is fast.

// TODO: Multi dimensional layers, so that it can be locally connected for 2D
// and 3D data.

// TODO: Assert DataType is numeric.

// TODO: Use AA for named parameters. Maybe with a mixin.

// TODO: Document correctly, with ddoc or something.
