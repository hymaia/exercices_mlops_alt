import pandas as pd
import numpy as np
from mlops_exo.features.markdown import add_total_markdown


# TODO 3.2 : tester la fonction add_total_markdown avec toutes les valeurs définies

def test_add_total_markdowns_all_values():
    # Given : input markdowns
    data = {'MarkDown1': [1], 'MarkDown2': [2], 'MarkDown3': [3], 'MarkDown4': [4], 'MarkDown5': [5]}
    df = pd.DataFrame(data)

    # When
    result = add_total_markdown(df)

    # Then
    expected = pd.DataFrame({'MarkDown1': [1], 'MarkDown2': [2], 'MarkDown3': [3], 'MarkDown4': [4], 'MarkDown5': [5],
                             'MarkdownsSum': [15]})
    assert result.equals(expected)

# TODO 3.2 : tester la fonction add_total_markdown avec seulement 3 colonnes markdown dispos


def test_add_total_markdowns_with_missing_markdowns():
    # Given : input markdowns
    data = {'MarkDown1': [1], 'MarkDown2': [2], 'MarkDown3': [3], 'MarkDown4': [4]}
    df = pd.DataFrame(data)

    # When
    result = add_total_markdown(df)

    # Then
    expected = pd.DataFrame({'MarkDown1': [1], 'MarkDown2': [2], 'MarkDown3': [3], 'MarkDown4': [4],
                             'MarkdownsSum': [10]})

    assert result.equals(expected)