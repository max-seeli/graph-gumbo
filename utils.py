def results_to_latex(results_df):
    """
    Convert a results dataframe to a latex table.
    
    The latex table should have the following structure:
    - toprule
    - Graph Products
    - Each factor graph (with category rotatet 90 degrees)
    
    The graph products and factor graphs should be variable depending on the results_df.

    The results_df should have the following structure:
    - index: factor graph
    - columns: graph products
    """
    columns = results_df.columns
    index = results_df.index

    graph_types = {
        'K': 'Complete',
        'P': 'Path',
        'S': 'Star',
    }

    graphs_per_type = {v: [] for v in graph_types.values()}

    for factor_graph in index:
        graph_type = graph_types[factor_graph[0]]
        graphs_per_type[graph_type].append(factor_graph)


    best_results = results_df.idxmin(axis=0)
       

    latex = '\\begin{tabular}{c' + '|r' * len(columns) + 'r}\n'
    latex += '\t\\toprule\n'
    latex += '\t& \\multicolumn{' + str(len(columns)) + '}{c}{Graph Products} & \\\\\n'
    multicol_columns = ['\\multicolumn{1}{c}{' + c + '}' for c in columns]
    latex += '\tFactor & ' + ' & '.join(multicol_columns) + ' & \\\\\n'

    for graph_type, factor_graphs in graphs_per_type.items():
        latex += '\t\\addlinespace[0.5ex]\n'
        latex += '\t\\cline{1-' + str(len(columns) + 2) + '}\n'
        latex += '\t\\addlinespace[0.5ex]\n'
        for i, factor_graph in enumerate(factor_graphs):
        
            str_results = []
            for product in columns:
                if factor_graph == best_results[product]:
                    str_results.append('\\textbf{' + str(results_df.loc[factor_graph, product]) + '}')
                else:
                    str_results.append(str(results_df.loc[factor_graph, product]))

            latex += '\t$' + factor_graph[0] + '_{' + factor_graph[1:] + '}$ & ' + ' & '.join(str_results) + ' & '
            if i == 0:
                latex += '\\multirow{' + str(len(factor_graphs)) + '}{*}{\\rotatebox[origin=c]{270}{' + graph_type + '}} '
            latex += '\\\\\n'
    latex += '\t\\bottomrule\n'
    latex += '\\end{tabular}'

    with open('results.tex', 'w') as f:
        f.write(latex)



    