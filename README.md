# Rubik's-Cube
A Rubik's Cube Simulator Made Through Unity

I exclusively did this project for people who put their efforts into Rubik's cube solving algorithm to test and debug their code. The simulator provides a platform to directly access the cube's current state over a press of a button. Meanwhile, it can also be used for general purposes, having a good time solving the cube independently.

The simulator is equipped with a cube solving algorithm named The Two-Phase Algorithm by [Herbert Kociemba](https://www.speedsolving.com/wiki/index.php/Herbert_Kociemba). It is an external package [Kociemba](https://github.com/Megalomatt/Kociemba).

The Fetch state button is exclusively meant for testing and debugging purposes and is only available for desktop users. The fetch state button saves the cube's current state into a text file named [CurrentState.txt](https://github.com/milind-prajapat/Rubiks-Cube/blob/main/Builds/Windows_x64/Rubiks%20Cube_Data/CurrentState.txt), which is available in the simulator's [data](https://github.com/milind-prajapat/Rubiks-Cube/tree/main/Builds/Windows_x64/Rubiks%20Cube_Data) directory.

## Instructions To Use
One can find the required files to run the simulator in the [Builds](https://github.com/milind-prajapat/Rubiks-Cube/tree/main/Builds) directory of the repository. Android users can use the [apk file](https://github.com/milind-prajapat/Rubiks-Cube/blob/main/Builds/Rubiks%20Cube.apk) to install the application, whereas windows users can go with the [Windows_x64](https://github.com/milind-prajapat/Rubiks-Cube/tree/main/Builds/Windows_x64) or [Windows_x86](https://github.com/milind-prajapat/Rubiks-Cube/tree/main/Builds/Windows_x86) directory, depending on their architecture.

You can use the pointing devices as well as the keyboard keys to use the simulator.

## Features
1. **Automation** enables the shuffling and solution of the cube
2. Integration with The **Two-Phase Algorithm** enables the cube to solve in the least number of steps
3. The ability to write the **cube's current state** into a text file enables the testing and debugging of the cube solving algorithms
4. **User-friendly UI**
5. **Multiple platform support**

## Key-Bindings
```
F - Front Face
L - Left Face
B - Back Face
R - Right Face
U - Up Face
D - Down Face

M - Centre-Right Vertical
S - Centre-Left Vertical
E - Centre Horizontal

X - Rotates Cube on R
Y - Rotates Cube on U
Z - Rotates Cube on F

P - Shuffle
Q - Solve
C - Get Current State

Uppercase turns 90° clockwise, and Lowercase turns 90° counter-clockwise
```

## References
1. [Windows Build](https://drive.google.com/file/d/1yPU8f04ILZ6PNuArOsjJkt3EMk4QQR_F/view?usp=sharing)
2. [Android Application](https://drive.google.com/file/d/1ExQ5nqQ2iixKeT88FqmeG6X6uyF3YHdG/view?usp=sharing)
