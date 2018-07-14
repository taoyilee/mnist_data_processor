# mnist_data_processor
MNIST IDX Data Processing Kit

This is a small tool kit to process raw MNIST database form [Yann Lecun's webpage](http://yann.lecun.com/exdb/mnist/)

## Quick start
1. Download 4 gzipped archives from [Yann Lecun's webpage](http://yann.lecun.com/exdb/mnist/)
    1. [train-images-idx3-ubyte.gz](http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz):  training set images (9912422 bytes)
    2. [train-labels-idx1-ubyte.gz](http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz):  training set labels (28881 bytes)
    3. [t10k-images-idx3-ubyte.gz](http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz):   test set images (1648877 bytes)
    4. [t10k-labels-idx1-ubyte.gz](http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz):   test set labels (4542 bytes)
2. Unzip above files with
    ```commandline
    gunzip train-images-idx3-ubyte.gz
    gunzip train-labels-idx1-ubyte.gz
    gunzip t10k-images-idx3-ubyte.gz
    gunzip t10k-labels-idx1-ubyte.gz
    
    ```     
3. Edit `config.ini` accordingly, brief explanations of each config entry are available in comments.
    ```ini
    [DEFAULT]
    data_dir = /home/tylee/01_dataset/MNIST ; The directory where your downloaded dataset resides
    t10k-images-idx3-ubyte = t10k-images-idx3-ubyte ; testing 10K image data filename (gunzip unzipped)
    t10k-labels-idx1-ubyte = t10k-labels-idx1-ubyte ; testing 10K image data filename (gunzip unzipped)
    train-images-idx3-ubyte = train-images-idx3-ubyte ; training 60K image data filename (gunzip unzipped)
    train-labels-idx1-ubyte = train-labels-idx1-ubyte ; training 60K label data filename (gunzip unzipped)
    ```
4. Install the downloaded package with pip
    ```commandline
    cd <path_of_cloned_repository>
    pip install .
    ```
    
## Author
Please contact the me (Michael (Tao-Yi) Lee) if there is something to inquire, regarding usage or suggestions on this package:

Email: taoyil AT UCI DOT EDU    

Your inputs are highly appreciated. :D

## License  
MIT License

Copyright (c) 2018 Michael (Tao-Yi) Lee

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
