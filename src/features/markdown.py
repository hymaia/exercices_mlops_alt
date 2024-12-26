
def add_total_markdown(df):
    list_markdown_cols = ['MarkDown%s' % i for i in range(1, 6)]
    df['MarkdownsSum'] = df[list_markdown_cols].sum(axis=1)
    return df
