﻿using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.Random;


namespace SharpNetwork
{
    internal class Program
    {
        public static void Main(string[] args)
        {
            NNetwork network = new NNetwork();
            network.Configure(2);
            var ex = new double[] {1, 2};
            var answ = new double[] {0.2};
            network.Train(ex, answ);
        }
        
    }
}