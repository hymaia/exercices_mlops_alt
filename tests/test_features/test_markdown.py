import pandas as pd
from mlops_exo.features.markdown import add_total_markdown


# TODO 3.2 : tester la fonction add_total_markdown avec toutes les valeurs définies


def test_add_total_markdowns_all_values():
    # Given : input markdowns
    df = pd.DataFrame(data={
        'MarkDown1': [10],
        'MarkDown2': [20], 
        'MarkDown3': [30],
        'MarkDown4': [40],
        'MarkDown5': [50]
    })
    expected_output = pd.DataFrame(data={'MarkDown1': [10],
                                          'MarkDown2': [20], 
                                          'MarkDown3': [30], 
                                          'MarkDown4': [40], 
                                          'MarkDown5': [50], 
                                          'MarkdownsSum': [150]})

    # When
    output = add_total_markdown(df)

    # Then
    pd.testing.assert_frame_equal(output, expected_output)


# TODO 3.2 : tester la fonction add_total_markdown avec seulement 3 colonnes markdown dispos
def test_add_total_markdowns_with_missing_markdowns():
    # Given : input markdowns - only 3 columns exist
    df = pd.DataFrame(data={
        'MarkDown1': [10],
        'MarkDown2': [20], 
        'MarkDown3': [30]
    })

    # When
    try:
        output = add_total_markdown(df)
        assert False, "Expected an error but function succeeded"
    except Exception as e:
        # Then - function should raise an error because MarkDown4 and MarkDown5 don't exist
        assert isinstance(e, KeyError)
        print(f"Error type: {type(e)}, Error message: {e}")
    
    # Answer : yes, the function should be rewritten to handle the case where some markdowns are missing
    pass
