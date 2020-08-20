## This code implements vanilla neural network and extensions, including double DQN, dueling DQN, priority experience buffer, and Canny edge detector wrapper
The purpose of this project is to design a deep neural network, incorporate various extensions and modify
hyperparameters to learn non-trivial Atari 2600 games. We first evaluate the system to learn Pong, and then
utilise the findings to run DQN learning on Breakout.

Our team was drawn to this particular project for several reasons. Foremost, the idea of training an AI to play
classic Atari games was appealing for the novelty, as our team is composed of several video game fans.
However a large attraction of the project was that it allows us to work with highly sophisticated Python packages
to construct complicated neural networks with relative ease. We are able to experience the process of
experimenting with learning networks and tuning hyperparameters in an advanced supervised system without
having to reinvent the wheel and code neural networks by hand.

## Our Team

- [Mariya Shmalko](https://github.com/et-cetra)
- [Sean Luo](https://github.com/lu0x1a0)
- [Joshua Derbe](https://github.com/JoshuaDerbe)
- [Arkie Owen](https://github.com/Arkington)

## Creating virtual environment/Setup

First you will need to create a virtual environment.
On Linux this is:

```
python3 -m venv env
```

Next, activate the virtual environment with:

```
source env/bin/activate
```

You will now be in the virtual environment. Upgrade pip if needed with:

```
pip install --upgrade pip
```

Then install the requirements file with:

```
pip install -r requirements.txt
```

You may also need to install ffmpeg and xvfb with:

```
sudo apt install ffmpeg xvfb
```

You will also need to run:
```
sudo apt-get install python-opengl
```

## Training

Make sure you are in the virtual environment with

```
source env/bin/activate
```

To run the training run:

```
python [breakout.py,pong.py] COMMAND ARGS
```

See the .py files for command argument meanings

Recordings should then appear in the /recording folder. Tensorboard diagrams appear in the /runs folder. Also .dat files will appear called [Pong,Breakout]NoFrameskip-v4-best.dat.

## Play trained model on environment

Note recordings are also available for viewing in vanilla-recordings and extension-recordings folder.
Also note, the models for vanilla version do not seem to perform as expected. 

Make sure you are in the virtual environment with:

```
source env/bin/activate
```

To run a specific trained model, run the following command:

```
python runTrained --model SAVED_MODEL_NAME.dat --mode MODE --env GAME_ENVIRONMENT
```

Example:
--mode c for cpu, g for gpu,
--env PongNoFrameskip-v4

Example command to run breakout extensions model: 

```
python runTrained.py -e BreakoutNoFrameskip-v4 -md BreakoutNoFrameskip-v4-best-extensions.dat
```

Windows also need to install ffmpeg and a window substitute for xvfb manually added to path.
