using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;


namespace SharpNetwork
{
    public class NNetwork : INetwork
    {
        private double learningRate = 0.1;
        private List<int> layersCounts = new List<int>();
        private List<Matrix<double>> weights = new List<Matrix<double>>();
        private List<Matrix<double>> biases = new List<Matrix<double>>();


        public int InputSize { get; private set; }

        public void Train(int[] trainingVec, int expectedOutput)
        {
            double[] input = trainingVec.Select(Convert.ToDouble).ToArray();
            Matrix<double> x = Vector<double>.Build.DenseOfArray(input).ToColumnMatrix();
            double[] output = {expectedOutput};
            Matrix<double> y = Vector<double>.Build.DenseOfArray(output).ToColumnMatrix();
            List<Matrix<double>> nabla_w = new List<Matrix<double>>();
            List<Matrix<double>> nabla_b = new List<Matrix<double>>();
            List<Matrix<double>> sums = new List<Matrix<double>>();
            List<Matrix<double>> activations = new List<Matrix<double>>();
            activations.Add(x);
            
            /*activating network remembering activation and summatory functions*/
            for (int i = 0; i < layersCounts.Count-1; i++)
            {
                var sum = weights[i] * activations.Last() + biases[i];
                sums.Add(sum);
                var activation = Utils.Sigmoid(sum);
                activations.Add(activation);
            }
            
            /*calculating error of the output layer*/
            var fin = activations.Last();
            Console.WriteLine("Network response: {0}", fin);
            Matrix<double> delta; 
            delta = (fin - y).PointwiseMultiply(Utils.SigmoidDerivative(sums.Last()));
            nabla_b.Insert(0, delta);
            Matrix<double> n_w;
            n_w = delta * activations[activations.Count - 2].Transpose(); 
            nabla_w.Insert(0, n_w);
            
            /*propagate back!*/
            for (int i = layersCounts.Count - 2; i > 0; i--)
            {
                delta = (weights[i].Transpose() * delta).PointwiseMultiply(Utils.SigmoidDerivative(sums[i-1]));
                nabla_b.Insert(0, delta);
                n_w = delta * activations[i - 1].Transpose();
                nabla_w.Insert(0, n_w);
            }
            
            /*updating weights and biases*/
            for (int i = 0; i < layersCounts.Count-1; i++)
            {
                biases[i] = biases[i] - nabla_b[i] * learningRate;
                Console.WriteLine("Biases{0}", i);
                Console.WriteLine(biases[i]);
                weights[i] = weights[i] - nabla_w[i] * learningRate;
                Console.WriteLine("Weights{0}", i);
                Console.WriteLine(weights[i]);
            }
            
        }
        
        

        public int Predict(int[] inputVec)
        {
            double[] input = inputVec.Select(Convert.ToDouble).ToArray();
            Matrix<double> activations = Vector<double>.Build.DenseOfArray(input).ToColumnMatrix();
            for (int i = 0; i < layersCounts.Count-1; i++)
            {
                activations = Utils.Sigmoid(weights[i] * activations + biases[i]);
            }
            return (int)Math.Round(activations.AsColumnMajorArray()[0]);
            /* here we guarantee, that output matrix is 1x1. Useless cumbersome int casts.*/
        }
        
        

        public void Configure(int inputSize)
        {
            if (inputSize < 1)
            {
                Console.WriteLine("Input layer must contain at least 1 unit");
                return;
            }
            this.InputSize = inputSize;
            layersCounts.Clear();
            layersCounts.Add(inputSize);
            if (inputSize > 5)
            {
                layersCounts.Add(inputSize);
            }
            else
            {
                layersCounts.Add(inputSize*2);
            }
            layersCounts.Add(1);
            BuildNetwork();
        }

        public void Reconfigure(int[] layers)
        {
            // TODO: Implement configuration of arbitrary layers count
        }

        private void BuildNetwork()
        {
            weights.Clear();
            biases.Clear();
            for (int i = 0; i < layersCounts.Count - 1; i++)
            {
                Matrix<double> layerWeights = Matrix<double>.Build.Random(layersCounts[i + 1], layersCounts[i]);
                weights.Add(layerWeights);
                Console.WriteLine("Layer {0} - {1} weights", i, i+1);
                Console.WriteLine(layerWeights);
                Matrix<double> layerBiases = Matrix<double>.Build.Random(layersCounts[i + 1], 1);
                biases.Add(layerBiases);
                Console.WriteLine("Layer {0} biases", i+1);
                Console.WriteLine(layerBiases);
            }
            Console.WriteLine("__________________ Network built successfull _______________________");
        }
    }
}