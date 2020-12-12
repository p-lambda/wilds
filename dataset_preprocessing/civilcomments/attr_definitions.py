ORIG_ATTRS = [
        'male',
        'female',
        'transgender',
        'other_gender',
        'heterosexual',
        'homosexual_gay_or_lesbian',
        'bisexual',
        'other_sexual_orientation',
        'christian',
        'jewish',
        'muslim',
        'hindu',
        'buddhist',
        'atheist',
        'other_religion',
        'black',
        'white',
        'asian',
        'latino',
        'other_race_or_ethnicity',
        'physical_disability',
        'intellectual_or_learning_disability',
        'psychiatric_or_mental_illness',
        'other_disability',
    ]

AGGREGATE_ATTRS = {
    'LGBTQ': [
        'homosexual_gay_or_lesbian',
        'bisexual',
        'other_sexual_orientation',
        'transgender',
        'other_gender'],
    'other_religions': [
        'jewish',
        'hindu',
        'buddhist',
        'atheist',
        'other_religion'
    ],
    'asian_latino_etc': [
        'asian',
        'latino',
        'other_race_or_ethnicity'
    ],
    'disability_any': [
        'physical_disability',
        'intellectual_or_learning_disability',
        'psychiatric_or_mental_illness',
        'other_disability',
    ],
    'identity_any': ORIG_ATTRS,
}

GROUP_ATTRS = {
    'gender': [
        'male',
        'female',
        'transgender',
        'other_gender',
    ],
    'orientation': [
        'heterosexual',
        'homosexual_gay_or_lesbian',
        'bisexual',
        'other_sexual_orientation',
    ],
    'religion': [
        'christian',
        'jewish',
        'muslim',
        'hindu',
        'buddhist',
        'atheist',
        'other_religion'
    ],
    'race': [
        'black',
        'white',
        'asian',
        'latino',
        'other_race_or_ethnicity'
    ],
    'disability': [
        'physical_disability',
        'intellectual_or_learning_disability',
        'psychiatric_or_mental_illness',
        'other_disability',
    ]
}
