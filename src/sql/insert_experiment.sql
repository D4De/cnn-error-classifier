INSERT INTO Experiment (
    batch_name, sub_batch_name, file_name, file_path, igid, bfm, N, H, C, W,
    spatial_class, spatial_class_params, value_classes_counts, domain_class,
    corrupted_values_count, corrupted_channels_count, layout, metadata
) VALUES (
    :batch_name, 
    :sub_batch_name, 
    :file_name,
    :file_path, 
    :igid, 
    :bfm, 
    :N, 
    :H, 
    :C, 
    :W, 
    :spatial_class, 
    :spatial_class_params, 
    :value_classes_counts, 
    :domain_class,
    :corrupted_values_count, 
    :corrupted_channels_count, 
    :layout, 
    :metadata);