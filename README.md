# Adaptive-Voting-Ensemble
An enhanced AdaBoost classifier using adaptive weak learner selection for improved classification accuracy.

Adaptive Voting Ensemble for Enhanced AdaBoost Performance

Overview
This project investigates the performance of AdaBoost ensemble classifiers using different weak learners, including Logistic Regression, Decision Trees, Neural Networks, and Plurality Voting. A novel Adaptive Voting Ensemble (Enhanced AdaBoost) is introduced, which dynamically selects the best weak learner for each iteration based on error minimization.

The study compares five ensemble models:
    AdaBoost-LR: AdaBoost with Logistic Regression as the weak learner.
    AdaBoost-DT: AdaBoost with Decision Tree as the weak learner.
    AdaBoost-NN: AdaBoost with Neural Network as the weak learner.
    AdaBoost-PV: AdaBoost with Plurality Voting among LR, DT, and NN.
    Enhanced AdaBoost (Adaptive Voting Ensemble): AdaBoost dynamically selecting the best weak learner at each iteration.

Key Features
✅ Implements multiple AdaBoost variants from scratch.
✅ Introduces an enhanced adaptive weak learner selection method.
✅ Evaluates performance on four real-world datasets.
✅ Performs statistical analysis using Paired t-tests to validate performance differences.
✅ Visualization with Line and Box plots to illustrate accuracy trends.

Dataset
The models are trained and evaluated on four datasets:

    Stroke_BRFSS (Health-related binary classification)
    Pulsar_star (Astronomy data for pulsar classification)
    Sepsis (Medical dataset for sepsis prediction)
    Skin_NonSkin (Image classification dataset for skin segmentation)

All datasets were preprocessed for balanced classes and reduced feature space to optimize computational efficiency.
Methodology

    Data Preprocessing:
        Splitting into 80% training and 20% testing.
        First three columns used as features, last column as label.

    Model Implementation:
        Base classifiers: Logistic Regression, Decision Trees, Neural Networks.
        Plurality Voting aggregates predictions from these classifiers.
        Enhanced AdaBoost dynamically selects the best weak classifier at each iteration.

    Evaluation:
        Accuracy comparison across varying weak learners (10-100 iterations).
        Performance benchmarking across four datasets.
        Paired t-tests conducted to measure statistical significance.

Results

Table: Accuracy of ensemble models on different datasets
Model	Stroke_BRFSS	Pulsar_star	Sepsis	Skin_NonSkin
AdaBoost-LR	0.553	0.903	0.528	1.000
AdaBoost-DT	0.715	0.900	0.485	0.998
AdaBoost-NN	0.522	0.927	0.492	1.000
AdaBoost-PV	0.605	0.917	0.512	1.000
Enhanced AdaBoost	0.722	0.907	0.527	1.000
Installation & Requirements

To run the project, install the required Python libraries:

pip install numpy pandas scikit-learn matplotlib seaborn statsmodels

How to Run
    Clone the repository:

git clone https://github.com/yourusername/Adaptive-Voting-Ensemble.git
cd Adaptive-Voting-Ensemble

Run the script:
    python EnhancedAdaboost.py

    This will:
        Train and evaluate all five AdaBoost variants.
        Generate accuracy results and visualization plots.

    View performance results:
        Line plot (line_plot.png): Accuracy vs Iterations.
        Box plot (box_plot.png): Accuracy distribution across models.
        Q-Q plot (QQ_plot.png): Residual analysis for model validity.

Future Improvements
📌 Implement more diverse weak learners (e.g., SVM, Random Forest).
📌 Optimize computational efficiency for large datasets.

Author
Ranbir Singh Kochar
MSc Data Science 2023-24, Lancaster University
🔗 www.linkedin.com/in/ranbir-singh-kochar-5303b6314
📧 Email: ranbirkochar@gmail.com
