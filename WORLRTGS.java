import java.util.*;

public class WORLRTGS {

    private int n;
    private int m;
    private double[] W;
    private List<Integer>[] preds;
    private double[] C;
    private int numHump;
    private int numIter;
    private double kmax;
    private Random rand = new Random();
    private List<Schedule> population;
    private Schedule leader;


    private double gamma = 0.9;

    public WORLRTGS(int n, int m, double[] W, List<Integer>[] preds, double[] C, int numHump, int numIter) {
        this.n = n;
        this.m = m;
        this.W = W;
        this.preds = preds;
        this.C = C;
        this.numHump = numHump;
        this.numIter = numIter;
        this.kmax = numIter;
        this.population = new ArrayList<>();
    }

    private static class Schedule {
        int n, m;
        int[] assignment;
        double fitness;
        double[] gpuLoad;

        Schedule(int n, int m) {
            this.n = n;
            this.m = m;
            this.assignment = new int[n];
            this.gpuLoad = new double[m];
        }

        Schedule clone() {
            Schedule s = new Schedule(n, m);
            System.arraycopy(this.assignment, 0, s.assignment, 0, n);
            System.arraycopy(this.gpuLoad, 0, s.gpuLoad, 0, m);
            s.fitness = this.fitness;
            return s;
        }

        void computeLoads(double[] W, double[] C) {
            Arrays.fill(gpuLoad, 0.0);
            for (int i = 0; i < n; i++) {
                int g = assignment[i];
                gpuLoad[g] += W[i] / C[g];
            }
        }

        double getLoadForTask(int i) {
            return gpuLoad[assignment[i]];
        }
    }

    private double computeFFT(Schedule s, double[] W, List<Integer>[] preds, double[] C) {
        double[] finish = new double[n];
        double[] gpuAvail = new double[m];
        Arrays.fill(gpuAvail, 0.0);
        for (int i = 0; i < n; i++) {
            double maxPredFinish = 0.0;
            for (int p : preds[i]) {
                maxPredFinish = Math.max(maxPredFinish, finish[p]);
            }
            int g = s.assignment[i];
            double gpuStart = gpuAvail[g];
            double start = Math.max(maxPredFinish, gpuStart);
            finish[i] = start + W[i] / C[g];
            gpuAvail[g] = finish[i];
        }
        s.fitness = Arrays.stream(finish).max().orElse(0.0);
        return s.fitness;
    }

    public void initialize() {
        population = new ArrayList<>();
        for (int h = 0; h < numHump; h++) {
            Schedule s = new Schedule(n, m);
            for (int i = 0; i < n; i++) {
                s.assignment[i] = rand.nextInt(m);
            }
            s.computeLoads(W, C);
            computeFFT(s, W, preds, C); // sets fitness
            population.add(s);
        }
        leader = population.get(0).clone();
        for (Schedule s : population) {
            if (s.fitness < leader.fitness) {
                leader = s.clone();
            }
        }
    }

    private Schedule generateNewSchedule(double[] hope, double[] W, double[] C) {
        Schedule s = new Schedule(n, m);
        double[] currentLoad = new double[m]; // current total processing time per GPU
        for (int i = 0; i < n; i++) {
            int bestA = 0;
            double minAbsDiff = Math.abs(hope[i] - (currentLoad[0] + W[i] / C[0]));
            for (int a = 1; a < m; a++) {
                double predLoad = currentLoad[a] + W[i] / C[a];
                double absDiff = Math.abs(hope[i] - predLoad);
                if (absDiff < minAbsDiff) {
                    minAbsDiff = absDiff;
                    bestA = a;
                }
            }
            s.assignment[i] = bestA;
            currentLoad[bestA] += W[i] / C[bestA];
        }
        s.computeLoads(W, C);
        return s;
    }

    public void run() {
        initialize();
        for (int k = 1; k <= numIter; k++) {
            double aPrime = 2.0 * (1.0 - (double) k / kmax);
            List<Schedule> newPop = new ArrayList<>();
            newPop.add(leader.clone()); // elitism: keep leader
            for (Schedule curr : population) {
                if (curr.fitness == leader.fitness) continue; // skip leader
                double r1 = rand.nextDouble();
                double A = 2.0 * aPrime * r1 - aPrime;
                double p = rand.nextDouble();
                double[] hope = new double[n];
                boolean useLeaderOrExploration = Math.abs(A) < 1.0;
                Schedule targetWhale;
                if (!useLeaderOrExploration) {
                    // Exploration: random whale
                    int randIdx = rand.nextInt(numHump);
                    while (population.get(randIdx).fitness == leader.fitness || population.get(randIdx) == curr) {
                        randIdx = rand.nextInt(numHump);
                    }
                    targetWhale = population.get(randIdx);
                } else {
                    targetWhale = leader;
                }
                double[] targetLoads = new double[n];
                for (int i = 0; i < n; i++) {
                    targetLoads[i] = targetWhale.getLoadForTask(i);
                }
                double[] currLoadsForTasks = new double[n];
                for (int i = 0; i < n; i++) {
                    currLoadsForTasks[i] = curr.getLoadForTask(i);
                }
                for (int i = 0; i < n; i++) {
                    double diff = targetLoads[i] - currLoadsForTasks[i];
                    double s = (rand.nextDouble() < 0.5 ? 1.0 : -1.0); // ±
                    double factor;
                    if (useLeaderOrExploration && p > 0.5) {
                        double l = 2.0 * rand.nextDouble() - 1.0; // [-1,1]
                        double b = 1.0;
                        double APrime = Math.exp(b * l) * Math.cos(2.0 * Math.PI * l);
                        factor = APrime;
                    } else {
                        factor = A;
                    }
                    hope[i] = targetLoads[i] + s * factor * diff;
                }

                Schedule newS = generateNewSchedule(hope, W, C);
                computeFFT(newS, W, preds, C);
                newPop.add(newS);
            }
            population = newPop.subList(0, numHump); // keep numHump best? or all, but limit to numHump
            if (population.size() > numHump) {
                population.sort(Comparator.comparingDouble(s -> s.fitness));
                population = new ArrayList<>(population.subList(0, numHump));
            }

            leader = population.get(0).clone();
            for (Schedule s : population) {
                if (s.fitness < leader.fitness) {
                    leader = s.clone();
                }
            }
        }

        System.out.println("Best FFT: " + leader.fitness);
    }


    public static void main(String[] args) {
        int n = 5;
        int m = 2;
        double[] W = {1.0, 2.0, 1.5, 3.0, 1.0};
        @SuppressWarnings("unchecked")
        List<Integer>[] preds = new List[n];
        for (int i = 0; i < n; i++) preds[i] = new ArrayList<>();
        // Simple DAG: 0->1, 0->2, 1->3, 2->3, 3->4
        preds[1].add(0); preds[2].add(0); preds[3].add(1); preds[3].add(2); preds[4].add(3);
        double[] C = {1.0, 1.0}; // homogeneous
        int numHump = 10;
        int numIter = 50;
        WORLRTGS worl = new WORLRTGS(n, m, W, preds, C, numHump, numIter);
        worl.run();
    }
}