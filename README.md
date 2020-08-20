# There code implements vanilla neural network and extensions, including double DQN, dueling DQN, priority experience buffer, and Canny edge detector wrapper
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
