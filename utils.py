def get_file_name_from_source_nodes(source_nodes):
    # Extract filenames
    filenames = []
    for node_with_score in source_nodes:
        node = node_with_score.node 
        if 'filename' in node.metadata:
            filenames.append(node.metadata['filename'])

    return filenames