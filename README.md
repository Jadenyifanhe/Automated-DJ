# A Masterpiece of All Music: A Machine Learning Approach to Automatic DJ

This repository contains the machine learning based automated DJ system developed for the final project of CMU
11-755/18-797 Machine Learning for Signal Processing by Yifan He, Yanqiao Wang, Yuchen Wu, and Tiancheng Zheng. The
codebase and README file are based on this [repository](https://github.com/aida-ugent/dnb-autodj) accompanying the
paper [*Vande Veire, Len, and De Bie, Tijl, "From raw audio to a seamless mix: creating an automated DJ system for Drum
and Bass." Journal Of Audio, Speech and Music Processing 2018, 13 (2018)*](https://doi.org/10.1186/s13636-018-0134-8).
The original system is developed in Python2 and is designed for drum and bass music only. We generalized the model to
general electronic dance music (EDM). The code for our system is developed in Python3.

## Installation and Environment Setup

The automated DJ system has been tested for Ubuntu 18.04 LTS and Python 3.9. It is recommended to install the automated
DJ system using `pip3` in a `conda` environment:

```
conda create -n "autodj" python=3.9.5
source activate autodj
pip install git+https://github.com/JosephZheng1998/Automated-DJ
```

In case the installation fails when installing `pyaudio`, perform the following commands and retry the installation:

```
sudo apt install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0
sudo apt install ffmpeg libav-tools
sudo pip3 install pyaudio
```

**Important: Windows and OSX users please install Ubuntu 18.04 LTS on a virtual machine because the `essentia` library
is difficult to install on system platforms other than Linux.**

## Running the Automated DJ System

Run the application with the following command:

`python3 -m autodj.main`

The application is controlled using commands. A typical usage would be as follows:

```
$ python3 -m autodj.main

>> loaddir /home/username/music/track
Loading directory "/home/username/music/track"...
30 songs loaded (0 annotated).
>> annotate
Annotating music in song collection...
...
Done annotating!
>> play
Started playback!
```

The following commands are available:

* `loaddir <directory>`: Add the `.wav` audio files in the specified directory to the pool of available songs.
* `annotate`: Annotate all the files in the pool of available songs that are not annotated yet. Note that this might
  take a while, and that in the current prototype this can only be interrupted by forcefully exiting the program (using
  the key combination `Ctrl+C`).
* `play`: Start a DJ mix. This command must be called after using the `loaddir` command on at least one directory with
  some annotated songs. Also used to continue playing after pausing.
* `play save`: Start a DJ mix, and save it to disk afterwards.
* `pause`: Pause the DJ mix.
* `stop`: Stop the DJ mix.
* `skip`: Skip to the next important boundary in the mix. This skips to either the beginning of the next crossfade, the
  switch point of the current crossfade or the end of the current crossfade, whichever comes first.
* `s`: Shorthand for the skip command
* `showannotated`: Shows how many of the loaded songs are annotated.
* `debug`: Toggle debug information output. This command must be used before starting playback, or it will have no
  effect.
* `stereo`: Toggle stereo audio support (enabled by default).

To exit the application, use the `Ctrl+C` key combination.
