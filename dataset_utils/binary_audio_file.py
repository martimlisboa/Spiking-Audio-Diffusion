import os
import numpy as np

class BinaryAudioFileObject():
    def __init__(self, binary_file_path, delete_if_existing=False):
        super(BinaryAudioFileObject, self).__init__()

        self._binary_file_path = binary_file_path

        if delete_if_existing:
            if os.path.exists(self.binary_file_path()):
                os.remove(self.binary_file_path())

        # Creation if non existing
        if not os.path.exists(self.binary_file_path()):
            f = open(self.binary_file_path(), 'wb+')
            f.close()

    def number_of_samples(self):
        return self.number_of_bytes_written() // self.size_of_int16()

    def binary_file_path(self):
        return self._binary_file_path

    def number_of_bytes_written(self):
        binary_file_path = self.binary_file_path()
        with open(binary_file_path, 'rb') as binary_file:
            binary_file.seek(0, 2)  # set cursor at the end
            size = binary_file.tell()
        return size

    def write_audio_data(self, audio_data_float):

        assert len(audio_data_float.shape) == 1

        binary_file_path = self.binary_file_path()
        binary_file_byte_start = self.number_of_bytes_written()

        audio_data_int16 = self.float_to_int16(audio_data_float)

        with open(binary_file_path, 'ab+') as binary_file:
            self.n_bytes = self.number_of_bytes_written() + self.size_of_int16() * audio_data_int16.size
            binary_file.write(audio_data_int16.astype(np.int16).tobytes())

        assert self.n_bytes == self.number_of_bytes_written(), "should be {} bytes and found {} in file ".format(self.n_bytes, self.number_of_bytes_written())
        binary_file_byte_end = self.number_of_bytes_written()

        binary_file_byte_length = binary_file_byte_end-binary_file_byte_start

        return binary_file_byte_start,binary_file_byte_length

    def max_of_int16(self):
        return 2 ** 15

    def int16_to_float(self,audio_data_as_int16):
        assert audio_data_as_int16.dtype == np.int16
        return audio_data_as_int16.astype(np.float32) / self.max_of_int16()

    def float_to_int16(self,audio_data_as_float):
        assert np.abs(audio_data_as_float).max() <= self.max_of_int16()
        return (audio_data_as_float * self.max_of_int16()).astype(np.int16)

    def size_of_int16(self):
        return 2

    def read_audio_data(self, sample_index_start, sample_index_end):
            size_of_sample = self.size_of_int16()

            assert sample_index_end >= sample_index_start
            assert size_of_sample * self.number_of_bytes_written() >= sample_index_end

            byte_index_start = size_of_sample * sample_index_start
            byte_index_end = size_of_sample * sample_index_end

            with open(self.binary_file_path(), 'rb') as binary_file:
                binary_file.seek(byte_index_start)
                audio_buffer = binary_file.read(byte_index_end - byte_index_start)
                audio_data_int16 = np.fromstring(audio_buffer, dtype=np.int16)

            audio_data_as_float = self.int16_to_float(audio_data_int16)
            return audio_data_as_float
