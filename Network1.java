import java.util.Scanner;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Network1
{
    protected int inputCount ;
    protected int NumberOfInputSets;
    protected double inputs[][];
    protected int layerCount;
    protected Layer1 Layers[];
    protected int outputCount;
    protected double outputs[];

    protected int NumberOfExpectedSets;
    protected double Expecteds[][];

    private int InOutCoherer = 0; 
    //Neuron1 controlNeuron = new Neuron1(); //Commented so that it can be replaced with Neuron1.learningrate = Arbitrary value; --21/05/2026--
    //Layer1 controlLayer = new Layer1(0);
    List<Double> Error_Plot = new ArrayList<>();
    protected int NeuronCountPerLayer[];
    Random R = new Random();

    Network1(int input,int output, int layer, int NCount[], int NoOfInputSets)
    {
        inputCount = input;
        layerCount = layer;
        outputCount = NCount[(NCount.length)-1];
        NumberOfExpectedSets = NoOfInputSets;
        NeuronCountPerLayer = NCount;
        inputs = new double[NoOfInputSets][inputCount];
        NumberOfInputSets = NoOfInputSets;
    }

    Network1()
    {

    }

    public static void main(){

        Network1 n = new Network1();
        n.initialize();
    }

    protected void initialize(){
        Expecteds = new double[NumberOfExpectedSets][outputCount];
        inputs = new double[NumberOfInputSets][inputCount];
        Layers = new Layer1[layerCount];
        outputs = new double[outputCount];
        try
        {
            for(int i = 0; i< layerCount; i++)
            {
                Layers[i] = new Layer1(NeuronCountPerLayer[i]);          
                Layers[i].initialize();
            }
        }
        catch(Exception e)
        {
            System.out.println(e);
        }

    }

    protected void accept(){
        Scanner sc = new Scanner(System.in);
        System.out.println("Enter those "+inputCount+" inputs: ");
        for(int i =0; i<NumberOfInputSets; i++)
        {
            System.out.println("Input Set " + i + " :::");
            for(int j =0; j<inputCount; j++)
            {
                inputs[i][j] = sc.nextDouble();
            }
            System.out.println("Expected Set " + i + " :::::");
            for(int j =0; j<outputCount; j++)
            {
                System.out.println("Enter the Expected Output: ");
                Expecteds[i][j] = sc.nextDouble();
            }
        }
    }

    protected void predict(double inps[])
    {
        // For first layer
        Layers[0].accept(inps);
        Layers[0].comput();

        //For all hidden layers
        for(int i = 1; i< layerCount; i++)
        {
            Layers[i].accept(Layers[i-1].results());
            Layers[i].comput();
        }

        // //For the output layer // commented because ouput layer is already taken care of in previous for() loop --02/11/2024--
        // Layers[layerCount-1].accept(Layers[layerCount - 2].results());
        // Layers[layerCount - 1].comput();

        for(int i = 0; i< Layers[layerCount - 1].results().length; i++)
        {
            //sumOfObtained +=  Layers[layerCount - 1].results()[i]; // Commented because it was the wrong approach --20/05/2026--
            System.out.println(i + " ___ " + Layers[layerCount - 1].results()[i]);
        }

    }

    protected void Backpro()
    {
        // For output layer --10/03/25--
        Layers[layerCount-1].Backprop();

        // For input layer --10/03/25--
        for(int i = layerCount-2 ; i>= 0 ; i--)
        {
            //Layers[i].Backprop_Hidden(Layers[i+1]);
            Layers[i].Backprop(Layers[i+1]);
        }
    }

    protected void BTBackpro(boolean Cycle_complete) // Added BTBackprop for BTTrain
    {
        Layers[layerCount-1].BTBackprop(Cycle_complete);

        for(int i = layerCount-2 ; i>= 0 ; i--)
        {
            //Layers[i].Backprop_Hidden(Layers[i+1]);
            Layers[i].BTBackprop(Layers[i+1], Cycle_complete);
        }
    }

    
    //SGD(Stochastic Gradient Descent) Training
    protected void SGDTrain(int epochs, double LearningRate)
    {
        double errorArray[] = new double[outputCount];
        Neuron1.learningRate = LearningRate;
        double Combined_error_for_graph = 0;

        for(int i = 0; i<epochs; i++)
        {
            Combined_error_for_graph = 0;
            // predict(inputs[i % NumberOfInputSets]); // Commented because it gave network the inputs in an order. --21/05/2026--
            //This meant that the learning of network will heavily depend on the order in which different input setts were entered. --21/05/2026--
            //The network didn't even learnt sometimes. (Tested experimentally)  --21/05/2026--
            InOutCoherer = R.nextInt(NumberOfInputSets);
            predict(inputs[InOutCoherer]);

            for(int j =0; j<outputCount; j++) //commented --02/11/2024--
            {
                errorArray[j]=( Expecteds[InOutCoherer][j]- Layers[layerCount - 1].results()[j] );  //commented --02/11/2024--
                Layers[layerCount - 1].Neurona[j].delta = errorArray[j];
                System.out.println("Epoch: "+i+" Error:"+j+"  "+errorArray[j]);  //--26/10/2024--
                Combined_error_for_graph += MSE(errorArray[j]);
            }
            //Backpropagation
            Backpro();  // --26/10/2024--

            //controlNeuron.error = MSE(er_ror);  //--26/10/2024--
            //Backpropagation
            //Backpro();

            Error_Plot.add(Combined_error_for_graph/outputCount);
        }

        GraphPanel P = new GraphPanel();
        P.createAndShowGui(epochs,  Error_Plot);
    }

    // Batch Training --22/05/2026--
    protected void BTTrain(int epochs, double LearningRate)
    {
        double errorArray[] = new double[outputCount];
        Neuron1.learningRate = LearningRate;
        double Combined_error_for_graph = 0;

        for(int i = 0; i<epochs; i++)
        {
            Combined_error_for_graph = 0;
            //The K loop is used to accumulate net change for all the inputs combined for each weight --24/05/2026--
            for(int k =0; k<NumberOfInputSets; k++)
            {
                // Here we can't use randomised inputs otherwise Batch training loses its essence.
                predict(inputs[k % NumberOfInputSets]);
                for(int j =0; j<outputCount; j++)
                {
                    errorArray[j]=( Expecteds[k % NumberOfInputSets][j]- Layers[layerCount - 1].results()[j] );
                    Layers[layerCount - 1].Neurona[j].delta = errorArray[j];
                    System.out.println("Epoch: "+i+" Error:"+j+"  "+errorArray[j]);
                    Combined_error_for_graph += ( MSE(errorArray[j]) )/NumberOfInputSets;
                }
                //Backpropagation to accumulate weight changes --24/05/2026--
                BTBackpro(false);
            }
            //Backpropagation to update weights --24/05/2026--
            BTBackpro(true);

            Error_Plot.add(Combined_error_for_graph);
            for(int j =0; j<outputCount; j++){
                Layers[layerCount - 1].Neurona[j].delta = 0; //Reset the delta for next epoch --23/05/2026--
            }
        }

        GraphPanel P = new GraphPanel();
        P.createAndShowGui(epochs,  Error_Plot);
    }

    public double MSE(double errors[]) //--26/10/2024--
    {
        double SquareSum = 0.0;
        for(int i = 0; i<errors.length; i++)
            SquareSum += Math.pow(errors[i],2);
        return ( SquareSum/errors.length);
    }

    public double MSE(double error) //--02/11/2024--
    {
        return (Math.pow(error, 2));
    }
}