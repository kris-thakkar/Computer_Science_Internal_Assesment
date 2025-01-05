import streamlit as st
import numpy as np
import time

import pandas as pd
import altair as alt
import matplotlib.pyplot as plt

from qiskit_aer import Aer
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import TwoLocal
from qiskit.visualization import circuit_drawer  # for direct circuit drawing

from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit.algorithms.optimizers import COBYLA

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


class QuantumXORClassifier:
    """
    Encapsulates the logic for:
      1. Generating & shuffling the XOR dataset
      2. Building the parametric quantum circuit (ansatz)
      3. Constructing an EstimatorQNN
      4. Training and evaluating a NeuralNetworkClassifier

    We also store the constructed circuit as a Matplotlib figure for display.
    """

    def __init__(
        self,
        test_size: float = 0.3,
        max_iters: int = 50,
        seed: int = 42,
        reps: int = 1,
        entanglement: str = 'full',
        shots: int = 1024
    ):
        """
        Initializes the classifier with user-defined hyperparameters.

        :param test_size: Ratio of the dataset to be used for testing
        :param max_iters: Maximum iterations for the COBYLA optimizer
        :param seed: Random seed for reproducibility
        :param reps: Repetitions of the entangling layer in TwoLocal ansatz
        :param entanglement: Entanglement pattern for TwoLocal
        :param shots: Number of shots (samples) for the quantum circuit measurement
        """
        self.test_size = test_size
        self.max_iters = max_iters
        self.seed = seed
        self.reps = reps
        self.entanglement = entanglement
        self.shots = shots

        # Placeholders for model state
        self.classifier = None
        self.X_test = None
        self.y_test_mapped = None
        self.y_pred = None
        self.accuracy = None
        self.classification_report = None

        # We'll store the final QuantumCircuit for optional display in the UI
        self.qc = None
        self.qc_figure = None

    def prepare_data(self):
        """
        Generates the synthetic XOR dataset and performs train-test split.
        """
        # 1. Synthetic XOR Dataset (2 features, binary labels)
        X_data = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ] * 10, dtype=float)  # repeated for a slightly bigger dataset

        # Map labels from {0,1} -> {-1,1}
        y_data = np.array([-1, 1, 1, -1] * 10)

        # Shuffle the dataset
        rng = np.random.default_rng(seed=self.seed)
        indices = rng.permutation(len(X_data))
        X_data, y_data = X_data[indices], y_data[indices]

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_data, y_data, test_size=self.test_size, random_state=self.seed
        )

        return X_train, X_test, y_train, y_test

    def build_classifier(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Builds the quantum circuit, QNN, and classifier, then fits the model to the training data.
        Also stores the circuit as a Matplotlib figure for easy Streamlit display.
        """

        # 2. Define Parametric Circuit with Data Embedding
        num_qubits = 2
        feature_dim = 2
        input_params = ParameterVector('x', length=feature_dim)

        qc = QuantumCircuit(num_qubits, name="ParametricCircuit")
        for i in range(num_qubits):
            qc.ry(input_params[i] * np.pi, i)

        # Add a trainable ansatz with user-selected repetitions & entanglement
        ansatz = TwoLocal(
            num_qubits,
            rotation_blocks=['ry'],
            entanglement_blocks='cz',
            reps=self.reps,
            entanglement=self.entanglement
        )
        qc.compose(ansatz, inplace=True)

        # Store the final circuit internally
        self.qc = qc

        # Create a Matplotlib figure of this circuit for Streamlit display
        self.qc_figure = qc.draw(output='mpl')  # returns a Matplotlib Figure

        # 3. Create an Estimator
        estimator = Estimator(
            options={
                'backend': Aer.get_backend('aer_simulator'),
                'shots': self.shots
            }
        )

        # 4. 2-qubit SparsePauliOp for Observables
        observables = [SparsePauliOp.from_list([('ZZ', 1.0)])]

        # 5. Build an EstimatorQNN
        qnn = EstimatorQNN(
            circuit=qc,
            observables=observables,
            input_params=input_params,
            weight_params=ansatz.parameters,
            estimator=estimator
        )

        # 6. Build the NeuralNetworkClassifier
        self.classifier = NeuralNetworkClassifier(
            neural_network=qnn,
            optimizer=COBYLA(maxiter=self.max_iters)
        )

        # Train the classifier
        self.classifier.fit(X_train, y_train)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        """
        Generates predictions on the test set and stores performance metrics.
        """
        # Classifier outputs in {-1,1}, convert them to {0,1}
        y_pred_raw = self.classifier.predict(X_test)
        self.y_pred = ((y_pred_raw + 1) / 2).astype(int).flatten()
        self.y_test_mapped = ((y_test + 1) / 2).astype(int)

        self.accuracy = accuracy_score(self.y_test_mapped, self.y_pred)
        self.classification_report = classification_report(
            self.y_test_mapped, 
            self.y_pred
        )

    def run_training_pipeline(self):
        """
        High-level method to:
         1. Prepare data
         2. Build & train the model
         3. Evaluate the results
        """
        X_train, self.X_test, y_train, y_test = self.prepare_data()
        self.build_classifier(X_train, y_train)
        self.evaluate(self.X_test, y_test)

    def get_results(self):
        """
        Returns the results of the classification in a convenient format.
        """
        return {
            "accuracy": self.accuracy,
            "classification_report": self.classification_report,
            "predictions": self.y_pred,
            "y_test_mapped": self.y_test_mapped,
            "X_test": self.X_test
        }
    
    def get_circuit_diagram(self):
        """
        Returns the Matplotlib figure representing the circuit.
        Useful for direct display in Streamlit.
        """
        return self.qc_figure


class QuantumXORApp:
    """
    Manages the Streamlit UI for a quantum XOR classifier, including:
      - Model hyperparameters
      - Interactive Altair charts
      - Actual quantum circuit diagram from Qiskit
    """

    def __init__(self):
        self._render_header()

    def _render_header(self):
        """
        Render the main header of the UI.
        """
        st.title(" ❄️ Quantum XOR Classifier Demo ❄️")
        st.markdown(
            """
            **Description**:  
            This app demonstrates a quantum-classical hybrid approach to classify 
            data generated by the XOR function, using a parameterized quantum circuit 
            and a QNN-based classifier from Qiskit's machine learning module.
            
            Configure your quantum circuit and optimizer settings in the **sidebar**, then
            click **Train Model** to see how well it classifies the XOR dataset.

            ---
            """
        )

    def _sidebar_configuration(self):
        """
        Create various sidebar controls for user-defined hyperparameters.
        """
        st.sidebar.header("Hyperparameters & Settings ⚙️")

        test_size = st.sidebar.slider(
            label="Test Size Ratio",
            min_value=0.1,
            max_value=0.9,
            value=0.3,
            step=0.05
        )

        max_iterations = st.sidebar.slider(
            label="COBYLA Max Iterations",
            min_value=10,
            max_value=200,
            value=50,
            step=10
        )

        random_seed = st.sidebar.number_input(
            label="Random Seed",
            min_value=0,
            value=42,
            step=1
        )

        # Additional variables
        st.sidebar.subheader("Quantum Circuit Parameters")
        reps = st.sidebar.slider(
            label="TwoLocal Repetitions",
            min_value=1,
            max_value=5,
            value=1
        )
        entanglement = st.sidebar.selectbox(
            label="Entanglement Pattern",
            options=["full", "linear", "sca", "circular"],
            index=0
        )
        shots = st.sidebar.number_input(
            label="Shots (samples)",
            min_value=100,
            max_value=20000,
            value=1024,
            step=100
        )

        return test_size, max_iterations, random_seed, reps, entanglement, shots

    def run(self):
        """
        Defines and runs the Streamlit UI, collecting user input
        and triggering the training/evaluation pipeline.
        """
        (test_size, max_iterations,
         random_seed, reps,
         entanglement, shots) = self._sidebar_configuration()

        st.write("Once you're ready, click below to train the QNN classifier on the XOR dataset.")

        if st.button("Train Model"):
            st.info("Initializing training pipeline...")
            progress_bar = st.progress(0)

            # Create classifier engine
            engine = QuantumXORClassifier(
                test_size=test_size,
                max_iters=max_iterations,
                seed=random_seed,
                reps=reps,
                entanglement=entanglement,
                shots=shots
            )

            # Step 1: Prepare data
            progress_bar.progress(20)
            st.info("1. Generating & splitting XOR dataset...")

            # Step 2: Build circuit & QNN, then train
            time.sleep(1.0)  # For visual effect
            st.info("2. Building quantum circuit and training EstimatorQNN...")
            progress_bar.progress(50)

            engine.run_training_pipeline()

            # Step 3: Evaluate results
            time.sleep(1.0)  # For visual effect
            st.info("3. Evaluating model performance...")
            progress_bar.progress(80)

            results = engine.get_results()

            # Done
            progress_bar.progress(100)
            st.balloons()
            st.success("Training Complete!")

            # Display results
            st.subheader("Results Summary")
            st.write(f"**Test Accuracy**: {results['accuracy']:.2f}")
            st.write("**Classification Report**:")
            st.text(results['classification_report'])

            st.write("**Predicted vs. Actual**:")
            st.write(f"**Predicted**: {results['predictions'].tolist()}")
            st.write(f"**Actual**:    {results['y_test_mapped'].tolist()}")

            # --- Circuit Diagram from Qiskit ---
            with st.expander("Show Quantum Circuit Diagram"):
                fig_circuit = engine.get_circuit_diagram()
                st.pyplot(fig_circuit)  # Display the Matplotlib figure

            # --- Create and display interactive graphs with Altair ---
            st.subheader("Interactive Plots (Altair): Actual vs. Predicted")
            df = pd.DataFrame({
                "X0": results["X_test"][:, 0],
                "X1": results["X_test"][:, 1],
                "Actual": results["y_test_mapped"],
                "Predicted": results["predictions"]
            })

            # Convert numeric {0,1} labels to strings for better color and shape distinction
            df["Actual_str"] = df["Actual"].astype(str)
            df["Predicted_str"] = df["Predicted"].astype(str)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Actual Labels**")
                chart_actual = (
                    alt.Chart(df)
                    .mark_circle(size=80)
                    .encode(
                        x="X0:Q",
                        y="X1:Q",
                        color=alt.Color("Actual_str:N", title="Label"),
                        shape=alt.Shape("Actual_str:N"),
                        tooltip=["X0", "X1", "Actual_str"]
                    )
                    .properties(
                        title="Actual Labels",
                        width="container",
                        height=400
                    )
                    .interactive()
                )
                st.altair_chart(chart_actual, use_container_width=True)

            with col2:
                st.markdown("**Predicted Labels**")
                chart_predicted = (
                    alt.Chart(df)
                    .mark_circle(size=80)
                    .encode(
                        x="X0:Q",
                        y="X1:Q",
                        color=alt.Color("Predicted_str:N", title="Label"),
                        shape=alt.Shape("Predicted_str:N"),
                        tooltip=["X0", "X1", "Predicted_str"]
                    )
                    .properties(
                        title="Predicted Labels",
                        width="container",
                        height=400
                    )
                    .interactive()
                )
                st.altair_chart(chart_predicted, use_container_width=True)


def main():
    """
    Main function to run the Streamlit app in an OOP manner.
    """
    app = QuantumXORApp()
    app.run()


if __name__ == "__main__":
    main()
