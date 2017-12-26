void main () {
    /+
    import neural;
    import activations;
    enum inputLen = 64;
    float [inputLen][64] testInputs;
    float [inputLen][64] testOutputs;

    foreach (ref ti; testInputs) {
        foreach (ref i; ti) {
            i = genRandom;
        }
    }
    foreach (ref to; testOutputs) {
        foreach (ref i; to) {
            i = genRandom;
        }
    }
    import dense, gru;
    auto nn = new NeuralNetwork ! (
        inputLen
        , float
        , genRandom // Can use both a function or a value as weight initialization.
        , Layer! (GRU) (inputLen)
        , Layer! (GRU) (inputLen)
        , Layer! (GRU) (inputLen)
        , Layer! (GRU) (inputLen)
        , Layer! (GRU) (inputLen)
        , Layer! (GRU) (inputLen)
        , Layer! (GRU) (inputLen)
        , Layer! (GRU) (inputLen)
    ); 


    static void meanSquaredError (A, B, C)
        (A output, in B expected, ref C accumulated) {
            accumulated [] += output [] - expected [];
        }

    import std.stdio;
    import std.datetime.stopwatch;
    auto del = () {
        nn.train!(
            `a/100`, meanSquaredError
            , false /* No stdout */
            , false /* Don't shuffle */
            , true  /* Parallel */
        )
        (
              256
            , 32 /* Batch size */
            , testInputs  []
            , testOutputs []
        ); 
    };
    auto time = benchmark!del (50);
    writeln (time);
    +/
}

float genRandom () {
    import std.random;
    return uniform (-0.01f, 0.01f);
}
