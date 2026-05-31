import java.util.*;
public class MatrixLayer{
    protected int NeuronCount;
    protected double NeuronMatrix[][];
    protected double BiasMatrix[];
    protected double Preactivation[];
    private double inputs[] ;

    protected double DeltaMatrix[];
    protected int OutputCount;
    private double outputs[];

    
    static double learningRate;
    //double err_array[];

    MatrixLayer(int I)  // --12/02/2024--
    {
        NeuronCount = I;
    }

    protected void initialize (int WeightCount)
    {
        OutputCount = NeuronCount;
        NeuronMatrix = new double[NeuronCount][WeightCount];
        BiasMatrix = new double[NeuronCount];
        Preactivation = new double[NeuronCount];
        outputs = new double[OutputCount];
        DeltaMatrix = new double[NeuronCount];
        for(int i =0; i< NeuronCount; i++)
        {
            for(int j=0; j<WeightCount; j++){
                System.out.println(i);
                NeuronMatrix[i][j] = (Math.random() - 0.5)* 0.1;
            }
            BiasMatrix[i] = (Math.random() - 0.5)* 0.1;  
            System.out.println(i);
        }
    }

    protected void accept(double[] a)
    {
        inputs = a;
    }

    protected void comput()
    {
        for(int i = 0; i< NeuronCount; i++)
        {
            //Here we declared a new variable preactiva to store preactivation temporarily. 
            //Why? So that we can escape the need to reset preactivation everytime. --26/05/2026--
            double Preactiva = 0.0;
            for(int j=0; j< NeuronMatrix[i].length; j++){
                Preactiva += NeuronMatrix[i][j] * inputs[j];
            }
            Preactiva += BiasMatrix[i];
            Preactivation[i] = Preactiva;
            outputs[i] = sigmoid(Preactivation[i]);
        }
    }

    protected double[] results()
    {
        return outputs;
    }
    // for output
    protected void Backprop()
    {
        for(int i = 0; i< NeuronCount; i++)
        {
            this.DeltaMatrix[i] = DeltaMatrix[i] * sigmoidderiv(Preactivation[i]);
            for(int j =0; j<NeuronMatrix[i].length; j++)
            {
                NeuronMatrix[i][j] += (1)* DeltaMatrix[i] * inputs[j] * learningRate;
            }
            BiasMatrix[i] += DeltaMatrix[i] * learningRate;
        }
    }
    // for hidden
    protected void Backprop(MatrixLayer L )
    {
        for(int i = 0; i< NeuronCount; i++)
        {
            DeltaMatrix[i] = 0;
            // Here L is the next layer a.k.a. the layer to which this neuron gives its output
            for(int j =0; j<L.NeuronCount; j++)
            {
                DeltaMatrix[i] += L.DeltaMatrix[j] * L.NeuronMatrix[j][i];
            }
            DeltaMatrix[i] = DeltaMatrix[i] * sigmoidderiv(Preactivation[i]);
            for(int k=0; k<NeuronMatrix[i].length; k++){
                NeuronMatrix[i][k] += inputs[k]* DeltaMatrix[i] * learningRate; 
            }
            BiasMatrix[i] += DeltaMatrix[i] * learningRate;

        }
    }
    
     private double sigmoid(double preactivation){
        double activated = 1/(1+Math.exp(-(preactivation)));
        return activated;

    }

    private double sigmoidderiv(double x)
    // here, give "preactivation" as the input for x
    {
        double sig_x = sigmoid(x); //Commented to optimize sigderiv --24/05/2026--
        double derivative = (sig_x)* (1 - sig_x);
        return derivative;
    }

    // //Backprops for Batch Training --23/05/2026--
    // protected void BTBackprop(boolean Ready)
    // {
    // for(int i = 0; i< NeuronCount; i++)
    // {
    // Neurona[i].BTbackpropagate_Out(Ready);
    // }
    // }
    // //Backprops for Batch Training --23/05/2026--
    // // for hidden
    // protected void BTBackprop(Layer1 L, boolean Ready)
    // {
    // for(int i = 0; i< NeuronCount; i++)
    // {
    // Neurona[i].BTbackpropagate_Hidden(i, L, Ready);
    // }
    // }

}