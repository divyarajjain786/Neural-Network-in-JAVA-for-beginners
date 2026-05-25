import java.util.*;
public class Layer1{
    protected int NeuronCount;
    protected Neuron1[] Neurona = new Neuron1[NeuronCount];
    private double inputs[] ;

    protected int OutputCount;
    private double outputs[];
    
    //double err_array[];

    Layer1(int I )  // --12/02/2024--
    {
        NeuronCount = I;
        OutputCount = NeuronCount;
    }

    protected void initialize ()
    {

        Neurona = new Neuron1[NeuronCount];

        outputs = new double[OutputCount];
        for(int i =0; i< NeuronCount; i++)
        {
            System.out.println(i);
            Neurona[i] = new Neuron1();
            System.out.println(i);
            Neurona[i].activate();
        }
        //err_array = new double[NeuronCount];
    }

    protected void accept(double[] a)
    {
        inputs = a;
    }

    protected void comput()
    {
        for(int i = 0; i< NeuronCount; i++)
        {
            // to be resolved -21/01/2024-
            outputs[i] = Neurona[i].compute(inputs);
        }
    }
    
    // protected void comput_First()  //commented on --01/1/2024--
    // {
        // for(int i = 0; i< NeuronCount; i++)
        // {
            // // to be resolved -21/01/2024-
            // outputs[i] = Neurona[i].compute(inputs[i]);
        // }
    // }

    protected double[] results()
    {
        return outputs;
    }
    // for output
    protected void Backprop()
    {
        for(int i = 0; i< NeuronCount; i++)
        {
            Neurona[i].backpropagate_Out();
            //err_array[i] = Neurona[i].delta;
        }
        //LayerDelta = MSE(err_array); // commented on --20/05/2026--
    }
    // for hidden
    protected void Backprop(Layer1 L )
    {
        for(int i = 0; i< NeuronCount; i++)
        {
            Neurona[i].backpropagate_Hidden(i, L);
            //err_array[i] = Neurona[i].delta;
        }
        //LayerDelta = MSE(err_array); //Comented cuz it was redundant and threw off the backpropagation --20/05/2026--
    }
    
    //Backprops for Batch Training --23/05/2026--
    protected void BTBackprop(boolean Ready)
    {
        for(int i = 0; i< NeuronCount; i++)
        {
            Neurona[i].BTbackpropagate_Out(Ready);
        }
    }
    //Backprops for Batch Training --23/05/2026--
    // for hidden
    protected void BTBackprop(Layer1 L, boolean Ready)
    {
        for(int i = 0; i< NeuronCount; i++)
        {
            Neurona[i].BTbackpropagate_Hidden(i, L, Ready);
        }
    }

}