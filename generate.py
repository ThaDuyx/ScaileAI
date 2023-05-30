import pickle
import numpy
# - music21 modules
from music21 import instrument, note, stream, chord

# - Keras modules
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import BatchNormalization as BatchNorm
from keras.layers import Activation

# - Flask modules
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

# - Initialise app
app = Flask(__name__)
CORS(app)

# - index route
@app.route("/")
def index():
    return "Welcome to the SMC2: AI Music Generator \n (The server is up and running!)"

# - Data fetching route
@app.route('/data')
def get_data():
    data = {'name': 'SMC02', 'description': 'AI Music Generator', 'version': '0.0.1' }
    return jsonify(data)

# - Sending midi file route
@app.route('/midi')
def send_midi():
    return send_file('generatedMIDI.mid', mimetype='audio/midi', as_attachment=True, download_name='generatedMIDI.mid')

@app.route('/generate')
def get_generate():
    generate()
    return send_file('generatedMIDI.mid', mimetype='audio/midi', as_attachment=True, download_name='generatedMIDI.mid')

@app.route('/download', methods=['POST'])
def download():
    data = request.get_json()
    key = data.get('key')
    scale = data.get('scale')

    response = {
        'key': key,
        'scale': scale
    }
    
    print(response)

    noteData, weight = selectWeightAndData(key, scale)

    print(noteData, weight)

    generate(noteData, weight)    

    return send_file('generatedMIDI.mid', mimetype='audio/midi', as_attachment=True, download_name='generatedMIDI.mid')
    

def selectWeightAndData(key, scale):
    if scale == "Random":
        print("Random selected")
        noteData = "data/midiChords"
        weight = "allChords16May.hdf5"
    else:
        print("Specific harmonic scale selected")

        noteData = "data/midiChords_cmaj"
        weight = "midi_chords_weight_cmaj.hdf5"

    return noteData, weight

# ------------------ GENERATE MUSIC ------------------
def generate(noteData, weight):
    # load the notes used to train the model
    #   - 'data/midiChords'
    #   - 'data/midiChords_cmaj'
    print("Loading notes...")
    with open(noteData, 'rb') as filepath:
        notes = pickle.load(filepath)

    # Get all pitch names
    pitchnames = sorted(set(item for item in notes))
    
    # Get all pitch names
    n_vocab = len(set(notes))

    print("Preparing sequences...")
    network_input, normalized_input = prepare_sequences(notes, pitchnames, n_vocab)
    print("Setting up network...")
    model = create_network(normalized_input, n_vocab, weight)
    print("Generating notes...")
    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab)
    print("Geneating MIDI...")
    create_midi(prediction_output)

def prepare_sequences(notes, pitchnames, n_vocab):
    # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    
    sequence_length = 100

    network_input = []
    output = []

    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)
 
    # reshape the input into a format compatible with LSTM layers
    normalized_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    normalized_input = normalized_input / float(n_vocab)

    return (network_input, normalized_input)

def create_network(network_input, n_vocab, weight):
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3,))
    model.add(LSTM(512))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    model.load_weights(weight)
    # model.load_weights('midi_chords_weight_cmaj.hdf5')
    # model.load_weights('allChords16May.hdf5')

    return model

def generate_notes(model, network_input, pitchnames, n_vocab):
    """ Generate notes from the neural network based on a sequence of notes """
    # - pick a random sequence from the input as a starting point for the prediction
    start = numpy.random.randint(0, len(network_input)-1)

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = network_input[start]
    prediction_output = []

    # - The range of the for loop is the number of notes you want to generate in the midi file
    for note_index in range(20):
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        index = numpy.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output

def create_midi(prediction_output):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    offset = 0
    output_notes = []

    # - create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        # Offset is the time between notes
        # If you want to have a longer duration between notes, increase the offset
        # Can be used in conjunction with the tempo to change the speed of the midi file
        offset += 2

    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp='generatedMIDI.mid')
    print("Done! MIDI file generated")

if __name__ == '__main__':
    app.run(port=5500)
    # generate()