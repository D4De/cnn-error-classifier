CREATE TABLE Experiment (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_name TEXT,
    sub_batch_name TEXT,
    file_name TEXT,
    file_path TEXT,
    igid TEXT,
    bfm TEXT,
    N INT NOT NULL,
    H INT NOT NULL,
    C INT NOT NULL,
    W INT NOT NULL,
    spatial_class TEXT,
    spatial_class_params JSON,
    spatial_pattern TEXT,
    value_classes_counts JSON,
    domain_class TEXT,
    corrupted_values_count INT,
    corrupted_channels_count INT,
    layout TEXT,
    metadata JSON
);
