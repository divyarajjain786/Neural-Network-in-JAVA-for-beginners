import java.util.Scanner;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class MatrixNetwork
{
    protected int inputCount ;
    protected int NumberOfInputSets;
    protected double inputs[][];
    protected int layerCount;
    protected MatrixLayer Layers[];
    protected int outputCount;
    protected double outputs[];

    protected int NumberOfExpectedSets;
    protected double Expecteds[][];

    private int InOutCoherer = 0;
    List<Double> Error_Plot = new ArrayList<>();
    protected int NeuronCountPerLayer[];
    Random R = new Random();
    
    MatrixNetwork(int input,int output, int layer, int NCount[], int NoOfInputSets)
    {
        inputCount = input;
        layerCount = layer;
        outputCount = NCount[(NCount.length)-1];
        NumberOfExpectedSets = NoOfInputSets;
        NeuronCountPerLayer = NCount;
        inputs = new double[NoOfInputSets][inputCount];
        NumberOfInputSets = NoOfInputSets;
    }

    MatrixNetwork()
    {

    }

    public static void main(){

        MatrixNetwork n = new MatrixNetwork();
        n.initialize();
    }

    protected void initialize(){
        Expecteds = new double[NumberOfExpectedSets][outputCount];
        inputs = new double[NumberOfInputSets][inputCount];
        Layers = new MatrixLayer[layerCount];
        outputs = new double[outputCount];
        //Initializing Layer0
        Layers[0] = new MatrixLayer(NeuronCountPerLayer[0]);          
        Layers[0].initialize(inputCount);
            
        try
        {
            for(int i = 1; i< layerCount; i++)
            {
                Layers[i] = new MatrixLayer(NeuronCountPerLayer[i]);          
                Layers[i].initialize(NeuronCountPerLayer[i-1]);
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

        for(int i = 0; i< Layers[layerCount - 1].results().length; i++)
        {
            System.out.println(i + " ___ " + Layers[layerCount - 1].results()[i]);
        }

    }

    protected void Backpro()
    {
        // For output layer --30/05/2026--
        Layers[layerCount-1].Backprop();

        // For input layer --30/05/2026--
        for(int i = layerCount-2 ; i>= 0 ; i--)
        {
            Layers[i].Backprop(Layers[i+1]);
        }
    }
    
    //SGD(Stochastic Gradient Descent) Training
    protected void SGDTrain(int epochs, double LearningRate)
    {
        double errorArray[] = new double[outputCount];
        Layers[0].learningRate = LearningRate;
        double Combined_error_for_graph = 0;

        for(int i = 0; i<epochs; i++)
        {
            Combined_error_for_graph = 0;
            predict(inputs[i % NumberOfInputSets]); // Commented because it gave network the inputs in an order. --21/05/2026--
            //This meant that the learning of network will heavily depend on the order in which different input setts were entered. --21/05/2026--
            //The network didn't even learnt sometimes. (Tested experimentally)  --21/05/2026--
            //InOutCoherer = R.nextInt(NumberOfInputSets);
            //predict(inputs[InOutCoherer]);

            for(int j =0; j<outputCount; j++) //commented --02/11/2024--
            {
                errorArray[j]=( Expecteds[i % NumberOfInputSets][j]- Layers[layerCount - 1].results()[j] );
                System.out.println("Epoch: "+i+" Error:"+j+"  "+errorArray[j]);
                Combined_error_for_graph += MSE(errorArray[j]);
            }
            Layers[layerCount - 1].DeltaMatrix = errorArray;
            //Backpropagation
            Backpro();  // --30/05/2026--
            Error_Plot.add(Combined_error_for_graph/outputCount);
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