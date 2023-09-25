# Speech Enhancement with ConvTasnet

This repository is dedicated to the training of a speech enhancement model using the ConvTasnet architecture. It offers flexibility by allowing you to experiment with different state-of-the-art loss functions defined in `Utils.py` and `Losses.py`. You can train the model for single-channel speech enhancement using `Tasnet.py`, or opt for a version adapted to multiple microphones using `model_Interchannel.py`.

## Getting Started

To get started, follow these steps:

1. **Choose Your Loss and Model Version**: Depending on your specific requirements, select the loss function and model version you want to use. These options are available in the respective files, `Utils.py` for utilities and `Losses.py` for loss functions.

2. **Prepare Your Dataset**: Ensure you have access to your speech dataset. You'll need the path to your speech dataset files for training.

3. **Room Impulse Responses (RIRs)**: If your application involves room impulse responses (e.g., for simulating real-world conditions), you'll also need the path to your room impulse response data.

4. **Launch the Training Script**: Run the training script using the following command:

   ```bash
   python Training.py
   ```
This script will start the training process based on your chosen loss function and model version. It will utilize the provided speech dataset and, if applicable, room impulse responses to train the ConvTasnet model for speech enhancement.




##  Additional Notes
. ** Feel free to modify the hyperparameters and training settings in the Training.py script to suit your specific needs.
. ** Ensure you have all the necessary dependencies and packages installed before running the training script.
. ** Monitor the training progress and check for any errors or warnings during the training process.


## Contact
For any questions or inquiries, please contact Mohammed HAFSATI @ hafsati.mohammed@gmail.com  