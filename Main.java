import java.util.Scanner;

/**
 * Write a description of class Main here.
 *
 * @author (your name)
 * @version (a version number or a date)
 */
public class Main
{
    Network1 N = new Network1();
    Scanner sc = new Scanner(System.in);
    
    public static void main(String[] args){
        Main M = new Main();
        M.start();
    }
    
    public void start()
    {
        System.out.println("Welcome to Neural Network");
        System.out.println("Choose desired options below:- \n 1. Train network on XOR \n 2. Train network on custom data(Mnaually enter data and decide architecture of network)");
        int a = sc.nextInt();
        switch(a)
        {
            case 1:
                XOR_Train();
                break;
            case 2:
                CustomTrain();
                break;
            default:
                System.out.println("Invalid Choice. Restart class.");
        }
    }

    public void CustomTrain(){
        int a,b,c,e;
        System.out.println("Enter number of inputs per set:");
        a = sc.nextInt();
        System.out.println("Enter number of outputs per set:");
        b = sc.nextInt();
        System.out.println("Enter number of Layers in the network:");
        c = sc.nextInt();
        int d[] = new int[c];
        for(int i = 0; i<b; i++){
            System.out.println("Enter number of neurons in layer" + i +" :");
            d[i] = sc.nextInt();
        }
        System.out.println("Enter number of input-output pairs in dataset:");
        e = sc.nextInt();
        N = new Network1(a,b,c,d,e);
        N.initialize();
        N.accept();
        System.out.println("Choose desired option: \n 1. Give input to check output \n 2. Train with stochastic gradient descent(Time heavy CPU light) \n 3. Train with Batch training(Time light CPU heavy) ");
        int h = sc.nextInt();
        switch(h){
            case 1:
                Forward();
                break;
            case 2:
                TRSGD();
                break;
            case 3:
                TRBT();
                break;
            default :
                System.out.println("Incorrect choice. Restart network. HAHAHAHAHAHAHAHA!!!!!");
        }

    }

    public void Forward(){
        double v[] = new double[N.inputCount];
        for(int i = 0; i<N.inputCount; i++)
            v[i] = sc.nextDouble();
        N.predict(v);
    }

    public void TRSGD(){
        System.out.println("Enter Learningrate:");
        double lr = sc.nextDouble();
        System.out.println("Enter Number of epochs:");
        int ep = sc.nextInt();
        N.SGDTrain(ep, lr);
    }

    public void TRBT(){
        System.out.println("Enter Learningrate:");
        double lr = sc.nextDouble();
        System.out.println("Enter Number of epochs:");
        int ep = sc.nextInt();
        N.BTTrain(ep, lr);

    }

    public void XOR_Train()
    {
        Initializer I = new Initializer();
        I.net();
        System.out.println("Choose desired option: \n 1. Train with stochastic gradient descent(Time heavy CPU light) \n 2. Train with Batch training(Time light CPU heavy) ");
        int h = sc.nextInt();
        switch(h){
            case 1:
                I.trSGD();
                break;
            case 2:
                I.trBT();
                break;
            default :
                System.out.println("Incorrect choice. Restart network. HAHAHAHAHAHAHAHA!!!!!");
        }
    }
}
