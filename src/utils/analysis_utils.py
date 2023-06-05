import pandas as pd

# %% obtener dataframe con el resultado del proceso
def organize_results(datasets): 
    id = []
    age = []
    sex = []
    hand = []
    outcome = []
    engel = []
    therapy = []
    implant = []
    target = []
    lesion_status = []
    age_onset = [] 

    for dataset in datasets:
        id.append(dataset.attrs['his_id'])
        age.append(dataset.attrs['age'])
        sex.append(dataset.attrs['sex'])
        hand.append(dataset.attrs['hand'])
        outcome.append(dataset.attrs['outcome'])
        engel.append(dataset.attrs['engel'])
        therapy.append(dataset.attrs['therapy'])
        implant.append(dataset.attrs['implant'])
        target.append(dataset.attrs['target'])
        lesion_status.append(dataset.attrs['lesion_status'])
        age_onset.append(dataset.attrs['age_onset'])

    dict(
        zip(id,
        zip(age,
        zip(sex,
        zip(hand,
        zip(outcome,
        zip(engel,
        zip(therapy,
        zip(implant,
        zip(target,
        zip(lesion_status,
        age_onset)))))))))))
    dictonary = {
        nombre_lista: lista for nombre_lista, lista in zip([
            'id',
            'age',
            'sex',
            'hand',
            'outcome',
            'engel',
            'therapy',
            'implant',
            'target',
            'lesion_status',
            'age_onset'],
            [id,
            age,
            sex,
            hand,
            outcome,
            engel,
            therapy,
            implant,
            target,
            lesion_status,
            age_onset])}

    df_process = pd.DataFrame(dictonary)
    df_process = df_process.set_index('id')
    return df_process