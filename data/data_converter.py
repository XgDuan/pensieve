import os
import sys
import numpy as np


FILE_PATH = sys.argv[1]
OUTPUT_PATH = sys.argv[2]
flag = bool(int(sys.argv[3]))
BYTES_PER_PKT = 1500.0
MILLISEC_IN_SEC = 1000.0
BITS_IN_BYTE = 8.0
B_IN_MB = 1000000.0

def main():
    for dir_path, dir_names, file_names in os.walk(FILE_PATH):
        for file_name in file_names:
            if flag:
                output_file_prefix = dir_path.split('\\')[-1].split('.')[0]
                output_file_name = 'report_' + output_file_prefix + '_' + '.'.join(file_name.split('.')[1:])
            else:
                output_file_name = file_name
            with open(os.path.join(dir_path, file_name), 'rb') as f, open(os.path.join(OUTPUT_PATH, output_file_name), 'wb') as mf:
                start_time = None
                for line in f:
                    parse = line.split()
                    if start_time is None:
                        start_time = float(parse[1]) / MILLISEC_IN_SEC
                    current_time = float(parse[1]) / MILLISEC_IN_SEC - start_time  # convert ms to s
                    current_bdwt = float(parse[4]) / (float(parse[5]) / MILLISEC_IN_SEC) / B_IN_MB
                    mf.write('%f %f\n' % (current_time, current_bdwt))

if __name__ == '__main__':
    main()
