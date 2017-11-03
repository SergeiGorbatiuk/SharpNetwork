
using System;
using MathNet.Numerics.LinearAlgebra;

namespace SharpNetwork
{
    public static class Utils
    {
        public static double Sigmoid(double val)
        {
            return 1 / (1 + Math.Exp(-val));
        }
        
        public static Matrix<double> Sigmoid (Matrix<double> vec)
        {
            return vec.Map((x) => Sigmoid(x));
        }

        public static double SigmoidDerivative(double val)
        {
            return Sigmoid(val) * (1 - Sigmoid(val));
        }

        public static Matrix<double> SigmoidDerivative (Matrix<double> vec)
        {
            return vec.Map((x) => SigmoidDerivative(x));
        }
    }
}