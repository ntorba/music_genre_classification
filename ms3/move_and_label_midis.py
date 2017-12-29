import os
import shutil


# this function moves files ending in specified ending from fromDir to toDir.
def sortFilesEnding(fromDir, toDir, ending):
    theList = os.listdir(fromDir) #list of files in directory to sort through
    for files in theList: 
        if files.endswith(ending): 
            shutil.move(os.path.join(fromDir, files), os.path.join(toDir, files)) #move the file from the fromDir to the toDir


#sortFilesEnding("/Users/nicholastorba/Downloads", "/Users/nicholastorba/trash", "pptx")

#sortFilesEnding("/Users/nicholastorba/Downloads", "/Users/nicholastorba/trash", "pptx")


def sortFilesBeginning(fromDir, toDir, beginning):
    theList = os.listdir(fromDir) #list of files in directory to sort through
    for files in theList: 
        if files.startswith(beginning): 
            shutil.move(os.path.join(fromDir, files), os.path.join(toDir, files)) #move the file from the fromDir to the toDir

midi_ending = '.mid'
sortFilesEnding('/Users/nicholastorba/Downloads', '/Users/nicholastorba/Machine_Learning/Midi_music_project/ed_midi', midi_ending)


# adds specified label to all songs that do not yet have a label

def add_labels(label, directory):
    all_labels=['cl_','cn_','ro_','hh_','mt_','ed_','pp_']
    file_list = os.listdir(directory)
    for file in file_list:
        if file[:3] not in all_labels:
            path = os.path.join(directory,file)
            target = os.path.join(directory, label+file)
            os.rename(path, target)

#this line would add the edm label to the front of any file that was not yet labeled. 
add_labels('ed_', '/Users/nicholastorba/Machine_Learning/Midi_music_project/ed_midi')

#return the number of songs with a specific label in a specified directory
def count_num_songs(label,directory):
    lst = os.listdir('/Users/nicholastorba/Machine_Learning/Midi_music_project/ed_midi')
    count = 0
    for i in lst:
        if (i[0:3]=='ed_'):
            count+=1
    return count

#uses count_num_songs to show how many edm files are in the file ed_midi
print(count_num_songs('ed_', '/Users/nicholastorba/Machine_Learning/Midi_music_project/ed_midi'))
    
#a_list = os.listdir('/Users/nicholastorba/Machine_Learning/Midi_music_project/ed_midi')
#count = 0
#for i in a_list:
#    if (i[0:2]=='ed'):
#        count+=1
#print(count)
