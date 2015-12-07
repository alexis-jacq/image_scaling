# image_scaling
===============

State-of-the-art image scaling, inspired by Directional Cubic Convolution Interpolation

![befor](doc/pap.jpg)
![after](doc/pap3.jpg)

![befor](doc/surf.png)
![after](doc/bigsurf.png)

### Installation

1 Download the repository :
```
$ git clone https://github.com/alexis-jacq/image_scaling.git
```

2 The library uses a standard ``CMake`` workflow:
```
$ mkdir build && cd build
$ cmake ..
$ make
```

### Usage
```
$ ./scaling picture_name scaled_picture_name
```
If you want to infinitly increase your resolution without hitting-eyes defaults you can use the 60 main clusters of colors :

```
$  python src/cluster.py -i picture_name -c 60
```
