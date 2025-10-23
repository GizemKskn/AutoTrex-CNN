# AutoTrex-CNN

This project aims to automatically play the T-Rex Runner game using a Convolutional Neural Network (CNN).

## Features

* Analyzes game screen images to recognize obstacles.
* Fast response using a trained CNN model (in `.h5` format).
* Easily runnable with Python (requirements listed in `requirements.txt`).

## Usage

1.  **Data Collection:** Run the `trex_getData.py` file to collect data from the game screen images.
2.  **Training:** Define and train your model with `trex_cnn.py`, or use a pre-trained model.
3.  **Automatic Play:** Start the automatic gameplay using the `trex_play.py` file.

## Requirements

* Python 3.x
* TensorFlow or Keras (for the CNN model)
* OpenCV (for image processing)
* Other dependencies are listed in `requirements.txt`.

## Contributing

If you wish to contribute to the project, you can open an issue or submit a pull request. Code comments should be kept clear and readable.

## License

This project is distributed under the MIT License — see the `LICENSE` file for details.
