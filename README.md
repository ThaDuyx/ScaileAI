# ScaileAI

<p align="center">
  <img width="200" src="https://github.com/ThaDuyx/Scaile/blob/main/Scaile/Supporting%20Files/Assets.xcassets/ScaileAI.imageset/ScaileAI.png?raw=true"
</p>

> A Python project used for creative purposes. The application generates chord progressions in the MIDI format through a LSTM recurrent neural network. The application is setup as a local server structure waiting for HTTP request. It works in conjunction with the Scaile app but can be expanded to a website environment.

- Developed in Python

## Purpose of the project
  - The purpose of the project was to explore if it was possible to generate chord progressions from a Neural Network. The object was to maintain the harmonic elements of the training data and generate a variety of chords with a sense of knowledge in music theory. Essentially matching the harmonic structure of a chord progression.
  

## Generating
  - By running the main.py project the training will start. This requires a path to a directory loaded with MIDI files. 
  - The training will first map the chords using the music21 framework resulting in a data structure of a list of chords.
  - The data will be mapped to a String format and converted to Integer for working with Keras.
  - When the training is done this will be converted back using music21 and streamed into a MIDI file.
  
## How To Use
<p align="center">
    <a href="http://www.youtube.com/watch?feature=player_embedded&v=TEudhz0Tdts" target="_blank">
    <img src="http://img.youtube.com/vi/TEudhz0Tdts/mqdefault.jpg" alt="Watch the video" width="400" height="220" border="10" />
    </a>
</p>
  


## Tools
### Data
- The training data consisted of a large bank of chord progression MIDI files made by the company _'Unison'_ which is listed in the project.

### Software
- Visual Studio Code

### Frameworks
- Music21
- Keras
- Flask

### iOS App
- iOS mobile application frontend: https://github.com/ThaDuyx/Scaile

### Supporting Files
- MIDI files (.mid)
