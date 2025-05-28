public class SRAMSimulator {
    private boolean[] memory;  // true = 1, false = 0

    public SRAMSimulator(int size) {
        memory = new boolean[size];  // starts with all 0s
    }
}