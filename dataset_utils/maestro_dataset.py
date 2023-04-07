import numpy as np
from torch.utils.data import Dataset
import torch
import torchaudio as T
import torchaudio.transforms as TT
import os
import pandas as pd

from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import wget
import zipfile
import librosa
#from dataset_utils.binary_audio_file import BinaryAudioFileObject
from .binary_audio_file import BinaryAudioFileObject
import matplotlib.pyplot as plt
import librosa.display as display
import pretty_midi



import random


class MaestroDataset(Dataset):


    def __init__(self, args):
        super(MaestroDataset,self).__init__()

        self.dataset_folder = args.dataset_folder
        self.seq_len = int(args.sequence_time*args.sample_rate)
        self.DATA_LIMIT = args.data_limit # put this to none in practice
        self.CLIP_LIMIT = args.clip_limit
        self.split = args.split
        self.data_set_name = "maestro-v3.0.0"
        self.sampling_rate = args.sample_rate
        self.midi_sampling_rate = args.midi_sampling_rate
        self.maestro_audio_url = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip"
        self.maestro_midi_url = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
        self.maestro_csv_url = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.csv"
        self.maestro_json_url = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.json"

        self.url_dict = {
            'audio_zip': self.maestro_audio_url,
            'midi_zip': self.maestro_midi_url,
            'csv': self.maestro_csv_url,
            'json': self.maestro_json_url
        }

        self.meta_data = pd.read_csv(self.get_local_path("csv"))
        self.audio_folder = os.path.join(self.dataset_folder,"audios")
        self.midi_folder = os.path.join(self.dataset_folder,"midis")

        if args.download and not self.is_data_extracted():
            if not self.are_zip_downloaded():
                for url_id, url in self.url_dict.items():
                    self.get_local_path(url_id, download=True)

            # extract all audio files
            if not os.path.exists(self.audio_folder): os.mkdir(self.audio_folder)
            print("Extracting audio files...")
            with zipfile.ZipFile(self.get_local_path("audio_zip"), 'r') as zip_ref:
                zip_ref.extractall(self.audio_folder)
            os.remove(self.get_local_path("audio_zip"))

            # extract all zip files
            if not os.path.exists(self.midi_folder): os.mkdir(self.midi_folder)
            print("Extracting midi files..")
            with zipfile.ZipFile(self.get_local_path("midi_zip"), 'r') as zip_ref:
                zip_ref.extractall(self.midi_folder)
            os.remove(self.get_local_path("midi_zip"))


        self.make_mono_pcm_files()
        self.index_table = []
        self.make_index_table()

        print(f"Instantiated MAESTRO dataset: {self.data_set_name}")
        print(f"    dataset folder  : {self.dataset_folder}")
        print(f"    split           : {self.split}")
        print(f"    sequence length : {self.seq_len} samples =  {self.seq_len/self.sampling_rate} s")
        print(f"    sampling rate   : {self.sampling_rate}")
        print(f"    midi sample rate: {self.midi_sampling_rate}")
        print(f"    download        : {args.download}")
        print(f"    data limit      : {self.DATA_LIMIT}")
        print(f"    clip limit      : {self.CLIP_LIMIT}\n")



    def get_local_path(self, url_id, download=True):
        local_path = self.get_local_path_of_url(self.url_dict[url_id])
        if not os.path.isfile(local_path) and download:
            print("Downloading {}...".format(url_id))
            url = self.url_dict[url_id]
            wget.download(url,local_path)

        return self.get_local_path_of_url(self.url_dict[url_id])

    def get_local_path_of_url(self,url):
        file_name = os.path.basename(url)
        return os.path.join(self.dataset_folder,file_name)

    def is_data_extracted(self):
        return os.path.exists(self.audio_folder) and os.path.exists(self.midi_folder)

    def are_zip_downloaded(self):
        audio_zip_exists = os.path.exists(self.get_local_path("audio_zip"))
        midi_zip_exists = self.get_local_path("midi_zip")
        return (audio_zip_exists and midi_zip_exists)

    def make_pcm_files(self):
        df = pd.read_csv(self.get_local_path("csv"))
        n_songs = df.shape[0]

        for i_song in range(0,n_songs):
            relative_file_path = df["audio_filename"][i_song]
            path = os.path.join(self.dataset_folder, "audios", self.data_set_name, relative_file_path)
            wav = librosa.core.load(path, mono=True)

    def is_valid_pcm_valid(self,audio_path, pcm_path):
        if not os.path.exists(pcm_path):
            return False

        # this is slightly hacky, but it's fast
        audio_duration = librosa.get_duration(filename=audio_path)
        pcm_obj = BinaryAudioFileObject(pcm_path,delete_if_existing=False)
        pcm_duration = pcm_obj.number_of_samples() / self.sampling_rate

        diff = np.abs(audio_duration - pcm_duration)
        #print("diff = ",diff)
        return diff < 1e-3

    def make_mono_pcm_files(self):
        print("Making Mono PCM files")
        n_songs = self.meta_data.shape[0]
        pcm_file_list = []

        for i_song in tqdm(range(n_songs)):
            relative_audio_path = self.meta_data["audio_filename"][i_song]
            relative_pcm_path = relative_audio_path[:-4] + "int16_mono_{}Hz.pcm".format(self.sampling_rate)
            audio_path = os.path.join(self.dataset_folder, "audios", self.data_set_name, relative_audio_path)
            pcm_path = os.path.join(self.dataset_folder, "audios", self.data_set_name, relative_pcm_path)
            #print(self.is_valid_pcm_valid(audio_path, pcm_path))
            if not self.is_valid_pcm_valid(audio_path, pcm_path):
                if self.DATA_LIMIT is None or i_song <= self.DATA_LIMIT:
                    binary_object = BinaryAudioFileObject(pcm_path,delete_if_existing=True)
                    data, _ = librosa.core.load(audio_path, mono=True, sr=self.sampling_rate)
                    binary_object.write_audio_data(data)

                    assert binary_object.number_of_samples() == data.size,\
                        "wrote {} samples but it had {}".format(binary_object.number_of_samples(), data.size)

                else:
                    relative_pcm_path = "null"


            pcm_file_list.append(relative_pcm_path)
        
        self.meta_data['pcm_filename'] = pcm_file_list
        self.meta_data.to_csv(self.get_local_path("csv"), index=False)

    def make_index_table(self):
        print("Making Index Table")
        index_table = []
        n_songs = self.meta_data.shape[0]
        n_samples_per_clip = self.seq_len

        count = 0;
        clip_limit = self.CLIP_LIMIT;
        if self.CLIP_LIMIT == None:
            clip_limit = count+1;
        for i_song in tqdm(range(n_songs)):
            if self.DATA_LIMIT is None or i_song < self.DATA_LIMIT:
                if self.split == self.meta_data["split"][i_song]:
                    relative_pcm_path = self.meta_data["pcm_filename"][i_song]
                    pcm_path = os.path.join(self.dataset_folder, "audios", self.data_set_name, relative_pcm_path)
                    pcm = BinaryAudioFileObject(pcm_path)

                    sample_start = 0
                    sample_end = sample_start + n_samples_per_clip
                    while sample_end < pcm.number_of_samples() and count<clip_limit :                        
                        indices = (i_song, sample_start, sample_end)
                        index_table += [indices]
                        sample_start += n_samples_per_clip
                        sample_end = sample_start + n_samples_per_clip
                        count +=1;
                        if self.CLIP_LIMIT == None:
                            clip_limit = count+1;

        self.index_table = index_table

    def time_to_string(self,t):
        t_seconds = int(t)
        t_minutes = t_seconds // 60
        t_hours = t_minutes // 60

        t_minutes = t_minutes % 60
        t_seconds = t_seconds % 60

        if t_hours == 0:
            return '{}:{:2d}'.format(t_minutes,t_seconds )
        else:
            return '{}:{}:{:2d}'.format(t_hours, t_minutes, t_seconds)

    def get_pcm_path(self,i_song):
        relative_pcm_path = self.meta_data["pcm_filename"][i_song]
        pcm_path = os.path.join(self.dataset_folder, "audios", self.data_set_name, relative_pcm_path)
        return pcm_path

    def parse_midi(self,i_song, sample_start, sample_end):
        t_start = sample_start / self.sampling_rate
        t_end = sample_end / self.sampling_rate

        midi_filename = self.meta_data['midi_filename'][i_song]
        midi_filepath = os.path.join(self.midi_folder, self.data_set_name, midi_filename)
        midi = pretty_midi.PrettyMIDI(midi_filepath)

        assert len(midi.instruments) == 1
        piano = midi.instruments[0]

        note_event_list = []
        control_change_64_event_list = []
        control_change_67_event_list = []

        for note in piano.notes:
            if note.start >= t_start and note.end <= t_end:
                note_start_index = int((note.start - t_start) * self.midi_sampling_rate)
                note_stop_index = int((note.end - t_start) * self.midi_sampling_rate)

                note_tuple = (note_start_index, note_stop_index, note.pitch, note.velocity)
                note_event_list.append(note_tuple)

        #for control_change in piano.control_changes:
        #    if t_start <= control_change.time <= t_end:
        #        assert control_change.number in [64, 67]
#
        #        control_time = int((control_change.time - t_start) * self.midi_sampling_rate)
        #        control_tuple = (control_time, control_change.value)
#
        #        if control_change.number == 64:
        #            control_change_64_event_list.append(control_tuple)
        #        elif control_change.number == 67:
        #            control_change_67_event_list.append(control_tuple)
        #        else:
        #            raise ValueError("Expected control_change number {}".format(control_change.number))
        return note_event_list#, control_change_64_event_list, control_change_67_event_list

    def get_number_of_midi_steps(self):
        clip_duration = self.seq_len / self.sampling_rate
        n_time = int(clip_duration * self.midi_sampling_rate) +1
        return n_time

    def make_midi_piano_roll(self, note_event_list):
        n_time = self.get_number_of_midi_steps()
        n_notes = 128

        piano_roll = np.zeros((n_time, n_notes, 2), dtype=np.int8)

        for (note_start, note_stop, pitch, velocity) in note_event_list:
            piano_roll[note_start,pitch,0] = velocity
            piano_roll[note_stop,pitch,1] = velocity

        return piano_roll

    def make_control_change_roll(self, control_change_64_event_list, control_change_67_event_list):
        n_time = self.get_number_of_midi_steps()

        control_change_roll = np.zeros((n_time, 2), dtype=np.int8)

        for (time, value) in control_change_64_event_list:
            control_change_roll[time, 0] = value

        for (time, value) in control_change_67_event_list:
            control_change_roll[time, 1] = value

        return control_change_roll

    def __len__(self):
        return len(self.index_table)        

    def __getitem__(self, idx):


        i_song, sample_start, sample_end = self.index_table[idx]

        pcm_path = self.get_pcm_path(i_song)
        pcm = BinaryAudioFileObject(pcm_path)
        audio_data = pcm.read_audio_data(sample_start, sample_end)

        metas = self.meta_data.to_dict(orient='records')[i_song]
        #note_event_list, control_change_64_event_list, control_change_67_event_list = \
        #self.parse_midi(i_song, sample_start, sample_end)

        #piano_roll = self.make_midi_piano_roll(note_event_list)
        #control_change_roll = self.make_control_change_roll(control_change_64_event_list, control_change_67_event_list)

        D = {
            'i_song': i_song,
            'audio': audio_data,
            'meta': metas,
            'sample_start': sample_start,
            'sample_end': sample_end,
            't_start': sample_start / self.sampling_rate,
            't_end': sample_end / self.sampling_rate
            #'note_events': note_event_list,
            #'control_64_changes': control_change_64_event_list,
            #'control_67_changes': control_change_67_event_list,
            #'piano_roll': piano_roll,
            #'control_change_roll': control_change_roll
            }

        return D
       

    def get_title(self,idx):
        i_song, _, _ = self.index_table[idx]
        metas = self.meta_data.to_dict(orient='records')[i_song]
        return f"{metas['canonical_composer']} - {metas['canonical_title']}"            


def from_maestro(args, is_distributed=False):
  dataset = MaestroDataset(args)
  #print(dataset.index_table)
  return torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=not is_distributed,
    collate_fn = Collator(args).collate_maestro,
    num_workers=os.cpu_count(),
    sampler=DistributedSampler(dataset) if is_distributed else None,
    pin_memory=True,
    drop_last=True
  )
class Collator:
    def __init__(self, args):
        self.args = args

    def collate_maestro(self,minibatch):
        audio = np.stack([record['audio'] for record in minibatch if 'audio' in record]) #Batch of audios
        return {
          'audio':torch.from_numpy(audio).unsqueeze(1), #Unsqueeze for mono channel
        }

         



if __name__ == "__main__":
    print("MAESTRO DATASET main")
