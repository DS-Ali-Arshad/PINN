# Solving the Black-Scholes Equation in Illiquid Markets Using Physics-Informed Neural Networks (PINNs)

## Description

This project implements a Physics-Informed Neural Network (PINN) to solve the **Black-Scholes equation for illiquid markets**. The spatial domain is \([0, 20]\) and the time domain is \([0, 0.25]\). 

The equation is solved with the following parameters:
- **Volatility** $\sigma = 0.2$
- **Liquidity parameter** $\rho = 0.02$
- **Risk-free rate** $r = 0.03$
- **Strike price** $E = 10$

The PINN approximates the solution to this partial differential equation (PDE) by minimizing the residual of the equation along with boundary and initial conditions. TensorFlow is used as the backend for the implementation.

---

## Requirements

To execute the code, the following libraries and their versions are required:

- Python >= 3.8  
- TensorFlow >= 2.4  
- NumPy >= 1.19  
- Matplotlib >= 3.3  

You can install the dependencies using the following command:

```bash
pip install tensorflow numpy matplotlib
```

---

## Code Structure

- **Define Equation and Problem Setup**: Specifies the Black-Scholes PDE, its parameters, and the boundary/initial conditions.
- **Data Preparation**: Generates collocation points and boundary data using TensorFlow tensors.
- **Model Initialization**: Defines a fully connected neural network (PINN) with scaling for input normalization.
- **Loss Function**: Combines PDE residual and boundary/initial loss terms.
- **Training**: Uses the Adam optimizer with a learning rate schedule to train the model for a specified number of epochs.
- **Visualization**: Plots:
  - The positions of collocation points.
  - The solution of the PDE in 3D.
  - The loss evolution over training epochs.

---

## Instructions to Run the Code

Follow these steps to repeat the experiment:

1. **Clone the Repository**:  
   Clone this repository to your local machine using:

   ```bash
   git clone https://github.com/DS-Ali-Arshad/PINN.git
   ```

2. **Set Up the Environment**:  
   Install the required Python libraries:

   ```bash
   pip install tensorflow numpy matplotlib
   ```

3. **Run the Code**:  
   Execute the Jupyter Notebook or the Python script to train the PINN and visualize the results:

   - If you have Jupyter Notebook installed:
     ```bash
     jupyter notebook
     ```
     Open the notebook file (`NLA_CP_PINN.ipynb`) and run all cells sequentially.

   - Alternatively, run the Python script (if converted):
     ```bash
     python NLA_CP_PINN.py
     ```

4. **Output**:  
   - **Collocation points**: A scatter plot showing the distribution of points used in training.  
   - **Solution surface**: A 3D plot of the predicted solution \( V(t, S) \).  
   - **Loss evolution**: A semilogarithmic plot showing the training loss over epochs.  

5. **Save Results**:  
   The final 3D plot of the solution will be saved as an image if `save_fig=True` is set in the `plot_soln` function.

---

## Results

The code generates:
- A 3D surface plot for the solution of the Black-Scholes PDE.  
- Training loss evolution demonstrating convergence over 4000 epochs.  

---

## References

1. **Gulen, Seda and Sari, Murat**  
   *"A Fréchet derivative-based novel approach to option pricing models in illiquid markets"*  
   *Mathematical Methods in the Applied Sciences*, 2022, Wiley Online Library.  

2. **Raissi, Maziar and Perdikaris, Paris and Karniadakis, George Em**  
   *"Physics informed deep learning (part I): Data-driven solutions of nonlinear partial differential equations"*  
   arXiv preprint arXiv:1711.10561, 2017.

3. **Blechschmidt, Jan and Ernst, Oliver G**  
   *"Three ways to solve partial differential equations with neural networks—A review"*  
   *GAMM-Mitteilungen*, 2021, Wiley Online Library.  
---

## Contact

For any questions or feedback, please contact:

- **Name: Ali Arshad, Hassan Iftikhar, Bogdan**  
- **Email**: [Ali.Arshad@skoltech.ru](mailto:Ali.Arshad@skoltech.ru)  
- **GitHub**: [https://github.com/DS-Ali-Arshad](https://github.com/DS-Ali-Arshad)
