from mne import set_bipolar_reference
import re

##convert monopolor montage to bipolar montage
def convert_to_bipolar(raw, drop_originals=True):
    original_ch_names = raw.ch_names
    ch_names_sorted = sorted(original_ch_names)
    ch_pairs = []
    for first, second in zip(ch_names_sorted, ch_names_sorted[1:]):
        firstName = re.sub(r'[0-9]+', '', first)
        secondName = re.sub(r'[0-9]+', '', second)
        if firstName == secondName:
            ch_pairs.append((first,second))
    for ch_pair in ch_pairs:
        raw = set_bipolar_reference(raw, ch_pair[0], ch_pair[1], drop_refs=False)
    if drop_originals:
        raw = raw.drop_channels(original_ch_names)
    return raw