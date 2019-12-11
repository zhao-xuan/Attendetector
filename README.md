# Attendetector
> *Important*: This project has not been finished(approximately 60% done). Current issues and improvements are listed at the end.
This is a nearly finished project on status detection based on facial expression recognition(See https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch). It is used for checking the status/availability of another person. This could be useful in private video chats or group conference.
> *Note 1*: Current Build only shows the result of the facial recognition detection, it needs further polish to detect whether a person is ***busy/might be busy/might be free/free***.
> *Note 2*: Pytorch can deploy <kbd>cuda()</kbd> in training model/claculating output. This program can automatically choose to use cuda if NVIDIA graphic card is available.

<img src="demo/demo.png" alt="Attendetector" style="width:256px;height:auto">

## Precondition ##
1. Linux system is required. MacOS support is coming soon.
2. You need to install following python dependencies:

        pip3 install opencv-python
        pip3 install torch
        pip3 install numpy
        pip3 install matplotlib --This is optional. You can use plot_result function and save the result, but you need to pull out the 'images' folder from 'pytorch_files'
        pip3 install scikit-image
        pip3 install pillow

## How to Use ##
1. Sender-Server-Client: run `python3 server_main.py` on server, run `python3 sender_main.py` on sender's side, and then run `python3 receiver_main.py` on receiver's side. Note you have to operate in the order as stated above, since <kbd>sender_main</kbd> still requires <kdb>server_main.py</kdb> to create server and port listener.
2. You can also choose to show the result at different stages. `plot_result` function allows you to save an image of output result with emojis, but you need to pull out the `images` folder in `pytorch_files`. You can also choose to show the frames by uncommenting codes in `run()` function in `sender_main.py`

## Current Build & Issues
1. Still need to optimize performance by chaging another neural network.
2. Still need to determine how to decide the status of the person
3. Still need to devise a front-end UI to operate properly