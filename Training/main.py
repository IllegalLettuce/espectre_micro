import numpy as np

def extract_features(csi_data):
    pass

def load_data():
    npz_baseline = np.load('baseline/baseline_c6_64sc_20251212_142443.npz')
    npz_baseline_noisy = np.load('baseline_noisy/baseline_noisy_c6_64sc_20251214_025208.npz')
    npz_baseline_movement = np.load('movement/movement_c6_64sc_20251212_142443.npz')
    # ['label', 'label_id', 'num_subcarriers', 'duration_ms', 'collected_at', 'format_version', 'csi_data', 'chip']


    print(F"{npz_baseline['label']} = {npz_baseline['label_id']}")
    print(F"{npz_baseline_noisy['label']} = {npz_baseline_noisy['label_id']}")
    print(F"{npz_baseline_movement['label']} {npz_baseline_movement['label_id']}")



    csi_data_example = npz_baseline['csi_data']
    print()
    print(csi_data_example[0])
    print(csi_data_example.shape)
    # [  0   0   0   0   0   0   0   0   0   0  25  10  25  10  27   9  30   8
    #   32   7  34   6  35   5  36   4  37   2  37   0  37  -2  36  -4  36  -5
    #   35  -7  34  -8  32 -10  29 -11  27 -12  24 -14  21 -15  18 -15  15 -16
    #   11 -16   7 -16   4 -16   0 -16  -3 -15  -7 -14 -11 -13 -14 -11 -18  -9
    #  -21  -8 -23  -6 -26  -4 -28  -1 -30   1 -32   3 -33   6 -34   8 -34  10
    #  -34  12 -34  15 -33  17 -32  18 -30  20 -28  21 -26  22 -24  23 -21  23
    #  -18  24 -16  23 -12  23  -9  23  -6  22  -3  20   1  18   4  17   7  14
    #    9  12]
    # (1000, 128)

def main():
    load_data()


if __name__ == '__main__':
    main()
















