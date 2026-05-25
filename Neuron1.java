import java.util.Scanner;
import java.util.*;
class Neuron1
{

    protected static double learningRate;
    protected double weights[] = new double[2];
    private double bias  = 1.0;
    private double output = 0.0;
    private double preactivation = 0.0;
    private double inputs[];

    double delta;
    double weight_delta_accu[] = new double[weights.length];
    double bias_accu = 0;
    //double deltos;

    //double error;
    protected void activate()
    {
        for(int i =0; i<weights.length; i++)
        {
            weights[i] = (Math.random() - 0.5)* 0.1; //Added this "-0.5" and "*0.1" specifically for XOR so that it may converge faster and better --21/05/2026--
        }
        bias = (Math.random() - 0.5) * 0.1; //Added this "-0.5" and "*0.1" specifically for XOR so that it may converge faster and better
        //err_array  = new double[weights.length + 1];
    }

    protected double compute(double in[])
    {
        inputs = in;
        // A new variable for preactivation had to be created so that, all the other preactivations from previous epochs would not spoil the training
        // Previously, the new value of precativation was just being added to the sum of all past preactivations as there was no mechanism to re-establish preactivation back to zero.
        //If we use the line "preactivation = 0.0;" , then it will hinder the backpropagation process. --12/02/2024--
        double preactiva = 0.0;
        for(int i =0; i<weights.length; i++)
        {
            preactiva += weights[i]*inputs[i];
        }
        preactiva += bias;
        preactivation = preactiva;
        output = sigmoid(preactivation);
        return output;
    }

    // protected double compute(double in) // Commented on --01/11/2024--
    // {
    // double in_to_inputs[] = {in};
    // inputs = in_to_inputs;
    // // A new variable for preactivation had to be created so that, all the other preactivations from previous epochs would not spoil the training
    // // Previously, the new value of precativation was just being added to the sum of all past preactivations as there was no mechanism to re-establish preactivation back to zero.
    // //If we use the line "preactivation = 0.0;" , then it will hinder the backpropagation process. --12/02/2024--
    // double preactiva = 0.0;

    // preactiva += weights[0]*inputs[0];

    // preactiva += bias;
    // preactivation = preactiva;
    // output = sigmoid(preactivation);
    // return output;
    // }

    private double sigmoid(double preactivation){
        double activated = 1/(1+Math.exp(-(preactivation)));
        return activated;

    }

    private double sigmoidderiv(double x)
    // here, give "preactivation" as the input for x
    {

        //double sig_x = sigmoid(x); //Commented to optimize sigderiv --24/05/2026--
        double derivative = (output)* (1 - output);
        return derivative;
    }

    private double tanh(double preactivation){
        double activated = (Math.exp(preactivation)-Math.exp(-preactivation))/(Math.exp(preactivation)+Math.exp(-preactivation));
        return activated;

    }

    private double tanhderiv(double x){
        double tanh_x = tanh(x);
        double derivative = 1 - (tanh_x)*(tanh_x);
        return derivative;
    }

    private double ReLu(double preactivation){
        double activated = Math.max(preactivation, 0.0);

        return activated;
    }

    private double ReLuderiv(double x){
        double derivative = x>0?1.0:0.0;

        return derivative;
    }

    protected void backpropagate_Out(){
        this.delta = delta * sigmoidderiv(preactivation);
        for(int i =0; i<weights.length; i++)
        {
            //weights[i] -= (inputs[i] * error * sigmoidderiv(preactivation)) * learningRate;
            weights[i] += (1)* delta * inputs[i] * learningRate;

        }

        bias += delta * learningRate;

        /**
         *  Delta to be added!
         */
        //delta = MSE(MSEderiv(Delta_Prev) * sigmoidderiv(preactivation)); //commented cuz it was throwing the whole concept of backporpagation of hidden layer --19/05/2026--
    }

    protected void backpropagate_Hidden(int neural_index, Layer1 L)
    {
        delta = 0;
        // Here L is the next layer a.k.a. the layer to which this neuron gives its output
        for(int i =0; i<L.Neurona.length; i++)
        {
            //weights[i] -= (inputs[i] * error * sigmoidderiv(preactivation)) * learningRate;
            //weights[i] += (-1)* MSEderiv(Delta_Prev) * sigmoidderiv(preactivation) * inputs[i] * learningRate;
            delta += L.Neurona[i].delta * L.Neurona[i].weights[neural_index] ;
        }
        delta = delta * sigmoidderiv(preactivation);
        for(int i=0; i<weights.length; i++){ //--19/05/2026--
            weights[i] += inputs[i]* delta * learningRate; 
        }
        bias += delta * learningRate;

        /**
         *  Delta to be added!
         */
        //delta = MSE(MSEderiv(Delta_Prev) * sigmoidderiv(preactivation)); //Removed becasue, real delta calculation was added above and this one was wrong --19/05/2026--
    }

    //Backpropagation for Batch training --23/05/2026--
    protected void BTbackpropagate_Out(boolean UpdateReady){
        if(UpdateReady){
            this.delta = delta * sigmoidderiv(preactivation);
            for(int i =0; i<weights.length; i++)
            {
                weight_delta_accu[i] += (1)* delta * inputs[i] * learningRate;
                weights[i] += weight_delta_accu[i];
            }
            bias_accu += delta * learningRate;
            bias += bias_accu;
        }
        else {
            this.delta = delta * sigmoidderiv(preactivation);
            for(int i =0; i<weights.length; i++)
            {
                weight_delta_accu[i] += (1)* delta * inputs[i] * learningRate;
            }
            bias_accu += delta * learningRate;
        }

    }

    //Backpropagation for Batch training --23/05/2026--
    protected void BTbackpropagate_Hidden(int neural_index, Layer1 L, boolean UpdateReady)
    {
        delta = 0;
        // Here L is the next layer a.k.a. the layer to which this neuron gives its output
        for(int i =0; i<L.Neurona.length; i++)
        {
            delta += L.Neurona[i].delta * L.Neurona[i].weights[neural_index] ;
        }
        delta = delta * sigmoidderiv(preactivation);
        
        if(UpdateReady){
            for(int i =0; i<weights.length; i++)
            {
                weight_delta_accu[i] += (1)* delta * inputs[i] * learningRate;
                weights[i] += weight_delta_accu[i];
            }
            bias_accu += delta * learningRate;
            bias += bias_accu;
            weight_delta_accu = new double[weights.length];
            bias_accu = 0;
        }
        else {
            for(int i =0; i<weights.length; i++)
            {
                weight_delta_accu[i] += (1)* delta * inputs[i] * learningRate;
            }
            bias_accu += delta * learningRate;
        }
    }

    public double MSE(double error) //--02/11/2024--
    {

        return (Math.pow(error, 2));
    }
    /**
    public double MSEderiv(double error) //-26/10/2024--
    {
    //Corrected the MSEderiv to its correct derivative --19/05/2026--
    return (-2 *(error));
    }
     */
    // protected void backpropagate_Hidden() //commented on --01/1/2024--
    // {
    // //protected double delta_for_this_neuron;
    // //protected double delta_for_other_neuron;

    // // for(int i =0; i<weightXdelta.length; i++)
    // // {
    // // delta += weightXdelta[i];
    // // }

    // deltos = delta;

    // for(int i =0; i<inputs.length; i++)
    // {
    // weights[i] -= (inputs[i] * deltos * sigmoidderiv(output)) * learningRate;
    // }
    // bias -= (deltos * sigmoidderiv(output)) * learningRate;

    // }

    // public static void main()
    // {
    // Neuron1 n = new Neuron1();
    // n.activate();
    // double exp[] = {1,0,1,0};
    // double ajgt[][] = {{1,0}, {0,0}, {0,1}, {1,1}} ;
    // for(int k = 0; k<7; k++)
    // for(int i =0; i<= 3; i++)
    // {
    // error = exp[i] - n.compute(ajgt[i]);
    // n.backpropagate();
    // System.out.println(n.compute(ajgt[i]) + " --- " + error);
    // }
    // }
}