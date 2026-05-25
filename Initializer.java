import java.util.Scanner;

public class Initializer
{
    Initializer(){
    }
    int a[] = {2,1};
    Network1 N = new Network1(2, 1, 2, a, 4);
    Scanner s = new Scanner(System.in);
    protected void net()//--29/10/2024--
    {
        N.initialize();
        double b[][] = {{0,0},{0,1},{1,1},{1,0}};
        N.inputs = b;
        double c[][] = {{0},{1},{0},{1}};
        N.Expecteds = c;
        N.initialize();
        //N.Layers[0].Neurona[0].weights[0] = 0.1;
        //N.Layers[0].Neurona[0].weights[1] = 0.02;
        //N.Layers[0].Neurona[1].weights[0] = 0.3;
        //N.Layers[0].Neurona[1].weights[1] = 0.7;
        //N.Layers[1].Neurona[0].weights[0] = 0.6;
        //N.Layers[1].Neurona[0].weights[1] = 0.9;
    }

    public void trSGD(){
        System.out.println("Enter Learningrate:");
        double lr = s.nextDouble();
        System.out.println("Enter Number of epochs:");
        int ep = s.nextInt();
        N.SGDTrain(ep, lr);

    
        
    }

    public void trBT(){
        System.out.println("Enter Learningrate:");
        double lr = s.nextDouble();
        System.out.println("Enter Number of epochs:");
        int ep = s.nextInt();
        N.BTTrain(ep, lr);
    }

}