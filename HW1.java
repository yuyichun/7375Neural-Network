public class HW1 {

    public static int activateNeuron(int[] inputs, double[] weights, double threshold) {
        double weightedSum = 0;

        for (int i = 0; i < inputs.length; i++) {
            weightedSum += inputs[i] * weights[i];
        }

        return weightedSum >= threshold ? 1 : 0;
    }

    public static void main(String[] args) {
        int[] inputs = { 1, 0, 1 }; 
        double[] weights = { 0.5, -0.2, 0.7 }; 
        double threshold = 0.5; 

        int output = activateNeuron(inputs, weights, threshold);
        System.out.println("The output of the McCulloch-Pitts neuron is: " + output);
    }
}
