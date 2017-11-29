import pretty_midi
import numpy as np
import itertools
import random
import glob

#Checks current directory for the midi file extension and returns a list of all the midi files
def getMidi():
    return glob.glob("Midi-Files\*.mid")

def chordTransform(chord): 
    #all possible triad chords
    triads = {
            'major' : [4, 3],
            'minor' : [3, 4],
            'dim' : [3, 3],
            'aug' : [4, 4]
        }
    
    #If not triad then returns a random note of the chord
    if len(chord) != 3:
        root_note = random.choice(chord)
        return root_note
    
    #Finds the corresponding notes and its root note
    root_chord = {}
    for note in chord:
        root_chord[note]= note%12
    
    # Get all possible permutations of these notes
    note_perms = list(itertools.permutations(list(root_chord.values())))

    # Test each permutation against the possible triad intervals and return the triad type if there's a match.
    for i in range(len(note_perms)-1):
        notes_intervals = []
        posRoot_note = 99
        root_note = 99

        # Loop through notes and create a list, length 2, of intervals to check against
        for j in range(len(chord)-1):
            
            #Stores the current and next note in the possible permutations
            note_A = note_perms[i][j]
            note_B = note_perms[i][j+1]
            
            #finds the interval
            interval = note_B - note_A
            
            #If the interval is negative then loops around just a different octave
            if interval < 0:
                interval = interval + 12
                
            #Store the interval
            notes_intervals.append(interval)
            
            #The lowest note is the possible root note so checks for that and stores it
            if note_A <= note_B:
                if note_A < posRoot_note:
                    posRoot_note = note_A
            if note_B <= note_A: 
                if note_B < posRoot_note:
                    posRoot_note = note_B
                    
        # Finally loop through the traids dict to see if we have a match for a triad
        for t in triads.keys():
            if triads[t] == notes_intervals:
                
                #If so the root note is the lowest note of the triad
                #This method finds a key given a value
                for real_root, pseudo_root in root_chord.items():
                    if pseudo_root == posRoot_note:
                        return real_root
            
    #If not then the root note is a random note from the collection of notes
    if root_note not in range(12):
        root_note = random.choice(list(root_chord.keys()))
        return root_note

#@inputs: note_array is a matrix that is 128xinstrument.get_piano_roll() long. The number of columns is dependent upon how
            # sample will be split by time

#@returns: a vector that contains the root note at each time sample
def instrument_to_vector(note_array):
    note_array_transpose = np.transpose(note_array)
    note_vector = np.empty(note_array.shape[1])
    note_vector.fill(-1)
    for i in range(note_array_transpose.shape[0]): #The i here will be the column number of the transpose, which is the note
                                                   #This loop should iterate through the number of columns in transpose
        note_list=[]
        for number in note_array_transpose[i]:
            if number!=-1:
                note_list.append(number) #add the number aka the note being played 
                                        # if there is no number there is no note played so that place is 0
            if len(note_list)!=1:
                note_vector[i]=-1
            else:
                note_vector[i]=note_list[0]
    return note_vector

def NoteMatrix(midi_data, samplesPerSec):
    #Defines how many samples per second
    fs = samplesPerSec

    #Returns the total amount of samples gotten
    y = np.arange(0, midi_data.get_end_time(), 1./fs).shape[0]

    #Our desired matrix has the amount of samples for every possible instrument
    #noteMatrix = np.zeros(shape=(128, y))
    noteMatrix = np.empty(shape=(128,y))
    noteMatrix.fill(-1)

    #Iterates through all the instruments of the midi song
    for instrument in midi_data.instruments:

        #Creates an array of all the notes the instrument can possibly play over a time sample and its velocity
        total_notes = np.asarray(instrument.get_piano_roll(fs=fs, times=np.arange(0, midi_data.get_end_time(), 1./fs)))
        total_notes[total_notes == 0] = -1
        
        #Holder for the final array that converts chords into notes making all instruments monophonic
        converted_notes = np.zeros(shape=total_notes.shape)

        #Goes through each time sample to see if notes repeat, if so find the root node of this chord
        i=0
        
        for column in total_notes.T:

            #Notes repeat in a time slice
            if count_nonNegOne(column) > 1:

                #create a list containing the notes played
                chord = np.where(column>=0)[0]
                
                if len(chord) > 0:
                    #finds the root note of the chord
                    root_note = chordTransform(chord)

                    #removes all other notes other than the root
                    for note in chord:
                        if note != root_note:
                            column[note] = -1

                    #Classify the time slice by the root note itself not velocity
                    column[root_note] = root_note

                #Store in the converted notes
            converted_notes[:, i] = column
            i += 1

        #As every time splice has only one note with the note defined, convert into vector
        instrument_vector = instrument_to_vector(converted_notes)

        #For that instrument store the vector of the notes played out of all
        noteMatrix[instrument.program] = instrument_vector
    return noteMatrix

def count_nonNegOne(array):
    count = 0
    for i in array:
        if i != -1:
            count += 1
    return count

def main():
    #Makes a list of all the note matrices for all midis
    midi_note = []

    #Makes a list of all the labels for each corresponding midi's note matrix
    midi_label = []

    #Iterates through all midis
    for midi in getMidi():

        #Opens midi as a pretty midi file
        midi_data = pretty_midi.PrettyMIDI(midi)

        #creates the note matrix
        noteMatrix = NoteMatrix(midi_data, 10)

        #adds to list of matrices
        midi_note.append(noteMatrix)

        #stores the label of the midi file which is the first two letters of each midi
        midi_label.append(midi[:2])
        
if __name__== "__main__":
    main()
