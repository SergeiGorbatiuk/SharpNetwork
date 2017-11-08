namespace SharpNetwork
{
    /// <summary>
    ///   Neural network inferface.
    /// </summary>
    public interface INetwork
    {
        /// <summary>
        ///   Number of neurons in input layer.
        /// </summary>
        int InputSize { get; }


        /// <summary>
        ///   Train network using training vector and expected output value.
        /// </summary>
        /// <param name="trainingVec">Training vector.</param>
        /// <param name="expectedOutput">Expected output value.</param>
        void Train(int[] trainingVec, int expectedOutput);


        /// <summary>
        ///   Get a predicted value from network using input vector.
        /// </summary>
        /// <param name="inputVec">Input vector.</param>
        /// <returns>Predicted value.</returns>
        int Predict(int[] inputVec);


        /// <summary>
        ///   Configure network to have specified number of neurons in input layer.
        /// </summary>
        /// <param name="inputSize">Number of neurons in input layer.</param>
        void Configure(int inputSize);
    }
}